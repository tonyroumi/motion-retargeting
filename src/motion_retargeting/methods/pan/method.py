from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from motion_retargeting.methods.base import RetargetingMethod
from motion_retargeting.resources.replay_buffer import ReplayBuffer
from motion_retargeting.data.visualization.visualize import visualize_motion

from .config import DomainSpec, PANConfig, PANModelConfig
from .kinematics import end_effector_velocity
from .losses import discriminator_loss, end_effector_terms, generator_adversarial_loss, mapped_root_velocity
from .model import PANModel
from .utils import (
    ForwardState,
    PairState,
    SourceState,
    require_finite,
    total,
    side_by_side,
    squared_error_over_height2,
    mpjpe,
)


class PANMethod(RetargetingMethod):
    """
    PAN retargeting method
    """

    def __init__(
        self,
        model_cfg: PANModelConfig,
        domain_specs: Dict[str, DomainSpec],
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
        cfg: PANConfig | None = None,
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.cfg = cfg if cfg is not None else PANConfig()
        self.domain_specs = domain_specs

        self.device = torch.device(device)

        self.model = PANModel(model_cfg, domain_specs).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.model.generator_parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator_parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )

        self.buffers = {
            domain: ReplayBuffer(
                self.cfg.replay_buffer_size,
                self.cfg.swap_probability,
            )
            for domain in self.domain_specs if not self.model.is_reference
        }

        self.num_steps = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        state = self.forward_state(batch)

        interval = self.cfg.loss_grad_norm_interval
        log_loss_grads = interval > 0 and self.num_steps % interval == 0
        self.num_steps += 1

        self.set_discriminator_requires_grad(False)
        generator_losses = self.compute_generator_losses(state)
        require_finite(generator_losses)
        generator_loss, generator_info = total(generator_losses, "total/generator")
        if log_loss_grads:
            generator_info.update(
                self.loss_grad_norms(generator_losses, list(self.model.generator_parameters()))
            )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        grad_info = self.grad_norms(("motion_encoder", "body_part_encoder", "decoder"))
        self.generator_optimizer.step()

        self.set_discriminator_requires_grad(True)
        discriminator_losses = self.compute_discriminator_losses(state)
        require_finite(discriminator_losses)
        total_discriminator_loss, discriminator_info = total(discriminator_losses, "total/discriminator")
        if log_loss_grads:
            discriminator_info.update(
                self.loss_grad_norms(discriminator_losses, list(self.model.discriminator_parameters()))
            )
        self.discriminator_optimizer.zero_grad()
        total_discriminator_loss.backward()
        grad_info.update(self.grad_norms(("discriminator",)))
        self.discriminator_optimizer.step()

        return {**generator_info, **discriminator_info, **grad_info}

    def grad_norms(self, components: tuple[str, ...]) -> Dict[str, float]:
        """
        L2 norm of the current parameter gradients of each (domain, component), keyed
        "grad/<domain>/<component>".
        """
        names, norms = [], []
        for domain, modules in self.model.domains.items():
            for component in components:
                grads = [p.grad for p in getattr(modules, component).parameters() if p.grad is not None]
                names.append(f"grad/{domain}/{component}")
                norms.append(
                    torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                    if grads else torch.zeros((), device=self.device)
                )
        return dict(zip(names, torch.stack(norms).tolist()))

    def loss_grad_norms(
        self,
        losses: Dict[str, torch.Tensor],
        parameters: List[nn.Parameter],
    ) -> Dict[str, float]:
        """
        L2 norm of each weighted loss term's gradient w.r.t. `parameters`, keyed
        "loss_grad/<term key>" (e.g. "loss_grad/group_a/position").
        """
        norms = []
        for loss in losses.values():
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            grads = [g for g in grads if g is not None]
            norms.append(
                torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                if grads else torch.zeros((), device=self.device)
            )
        return {f"loss_grad/{name}": norm for name, norm in zip(losses, torch.stack(norms).tolist())}

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        state = self.forward_state(batch)
        domains = self.model.domains
        if self.model.is_reference:
            return self.reference_validation_metrics(state)
        metrics = {}

        for name, source in state.sources.items():
            other = next(domain for domain in domains if domain != name)
            fk = domains[name].fk
            heights = domains[name].heights[source.bank_ids]

            positions = fk.world_positions(source.bones)
            reconstruction = fk.world_positions(source.reconstruction_bones)
            cycle = fk.world_positions(state.pairs[(name, other)].cycle_bones)

            prefix = f"{name}/val/"
            metrics.update({
                prefix + "retarget_latent_l1": F.l1_loss(
                    state.outputs.pairs[(name, other)].fake_code, state.outputs.motion_codes[name]
                ).item(),
                prefix + "retarget_realism": domains[other].discriminator(state.pairs[(name, other)].fake_bones).mean().item(),
                prefix + "reconstruction_mse": F.mse_loss(source.reconstruction, source.motion).item(),
                prefix + "reconstruction_mpjpe_m": mpjpe(reconstruction, positions),
                prefix + "reconstruction_squared_error_over_height2": squared_error_over_height2(reconstruction, positions, heights),
                prefix + "cycle_mpjpe_m": mpjpe(cycle, positions),
                prefix + "cycle_squared_error_over_height2": squared_error_over_height2(cycle, positions, heights),
            })

        return metrics

    def reference_validation_metrics(self, state: ForwardState) -> Dict[str, float]:
        metrics = {}
        for name, source in state.sources.items():
            domain = self.model.domains[name]
            selected = self.domain_specs[name].reference.position_indices
            target = domain.fk.world_positions(source.bones)[:, :, selected]
            other = next(key for key in self.model.domains if key != name)
            for label, bones in (("reconstruction", source.reconstruction_bones),
                                 ("cycle", state.pairs[(name, other)].cycle_bones)):
                positions = domain.fk.world_positions(bones)[:, :, selected]
                metrics[f"{name}/val/{label}_selected_mse_cm2"] = F.mse_loss(positions, target).item()
                metrics[f"{name}/val/{label}_selected_mpjpe_m"] = mpjpe(positions * .01, target * .01)
        return metrics

    def set_discriminator_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.model.discriminator_parameters():
            parameter.requires_grad_(requires_grad)

    def forward_state(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> ForwardState:
        outputs = self.model.forward_generator(batch)
        domains = self.model.domains

        sources = {}
        for name, domain in domains.items():
            bank_ids, motion = batch[name]["bank_id"], batch[name]["motion"]
            reconstruction = outputs.reconstructions[name]
            sources[name] = SourceState(
                bank_ids=bank_ids,
                motion=motion,
                denormalized=domain.denormalize(motion, bank_ids),
                bones=domain.bones(motion, bank_ids).detach(),
                reconstruction=reconstruction,
                reconstruction_denormalized=domain.denormalize(reconstruction, bank_ids),
                reconstruction_bones=domain.bones(reconstruction, bank_ids),
            )

        pairs = {}
        for (source, destination), pair in outputs.pairs.items():
            pairs[(source, destination)] = PairState(
                target_ids=pair.target_ids,
                fake_bones=domains[destination].bones(pair.fake, pair.target_ids),
                cycle_bones=domains[source].bones(pair.cycle, sources[source].bank_ids),
            )

        return ForwardState(outputs=outputs, sources=sources, pairs=pairs)

    def normalized_positions(self, domain: str, bones: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Root-relative positions with the absolute root entry, over each sample's height. """
        modules = self.model.domains[domain]
        return modules.fk.root_relative_with_absolute_root(bones) / modules.heights[bank_ids][:, None, None, None]

    def normalized_end_effector_velocity(self, domain: str, bones: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Scaled world end-effector displacement per frame, over each sample's height. """
        modules = self.model.domains[domain]
        velocity = end_effector_velocity(modules.fk.world_positions(bones), modules.ee_ids)
        return velocity / modules.heights[bank_ids][:, None, None, None]

    def compute_generator_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        if self.model.is_reference:
            return self.compute_reference_generator_losses(state)
        cfg = self.cfg
        losses = {}

        positions = {}
        velocities = {}
        # Reconstruction losses
        for name, source in state.sources.items():
            heights = self.model.domains[name].heights[source.bank_ids][:, None, None]
            positions[name] = self.normalized_positions(name, source.bones, source.bank_ids)
            velocities[name] = self.normalized_end_effector_velocity(name, source.bones, source.bank_ids)

            losses[f"{name}/reconstruction"] = cfg.reconstruction_lambda * F.mse_loss(source.reconstruction, source.motion)

            losses[f"{name}/position"] = cfg.position_lambda * F.mse_loss(
                self.normalized_positions(name, source.reconstruction_bones, source.bank_ids),
                positions[name],
            )

        # Retargeting
        for (source, destination), pair in state.pairs.items():
            route = f"{source}_to_{destination}"
            bank_ids = state.sources[source].bank_ids

            losses[f"{route}/latent_cycle"] = cfg.cycle_lambda * F.l1_loss(
                state.outputs.pairs[(source, destination)].fake_code,
                state.outputs.motion_codes[source],
            )
            losses[f"{route}/position_cycle"] = cfg.cycle_lambda * F.mse_loss(
                self.normalized_positions(source, pair.cycle_bones, bank_ids),
                positions[source],
            )

            # transfer, contact = end_effector_terms(
            #     velocities[source],
            #     self.normalized_end_effector_velocity(destination, pair.fake_bones, pair.target_ids),
            #     cfg.contact_threshold,
            # )
            # losses[f"{route}/end_effector"] = cfg.end_effector_lambda * transfer
            # losses[f"{route}/contact"] = cfg.end_effector_lambda * cfg.contact_lambda * contact

        adversarial = self.compute_adversarial_losses(state)
        losses.update({name: cfg.adversarial_lambda * loss for name, loss in adversarial.items()})

        return losses

    def compute_reference_generator_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        """norm_rec objective; cm FK, saved domain statistics, two cross routes only."""
        cfg, losses, positions = self.cfg, {}, {}
        for name, source in state.sources.items():
            domain, spec = self.model.domains[name], self.domain_specs[name]
            reference = spec.reference
            shape = (*source.motion.shape[:2], spec.num_joints, 4)
            selected = reference.rotation_indices
            losses[f"{name}/reconstruction"] = cfg.reconstruction_lambda * F.mse_loss(
                source.reconstruction[..., :-4].reshape(shape)[:, :, selected],
                source.motion[..., :-4].reshape(shape)[:, :, selected],
            )
            losses[f"{name}/root"] = cfg.reconstruction_lambda * cfg.root_lambda * F.mse_loss(
                source.reconstruction[..., -4:], source.motion[..., -4:],
            )
            positions[name] = domain.fk.world_positions(source.bones)[:, :, reference.position_indices]
            losses[f"{name}/position"] = cfg.reconstruction_lambda * cfg.position_lambda * F.mse_loss(
                domain.fk.world_positions(source.reconstruction_bones)[:, :, reference.position_indices], positions[name],
            )
        for (source, destination), pair in state.pairs.items():
            route = f"{source}_to_{destination}"
            output = state.outputs.pairs[(source, destination)]
            src_ref, dst_ref = self.domain_specs[source].reference, self.domain_specs[destination].reference
            losses[f"{route}/latent_cycle"] = cfg.cycle_lambda * F.mse_loss(output.fake_code, state.outputs.motion_codes[source])
            losses[f"{route}/position_cycle"] = cfg.cycle_lambda * F.mse_loss(
                self.model.domains[source].fk.world_positions(pair.cycle_bones)[:, :, src_ref.position_indices], positions[source],
            )
            original = state.sources[source].denormalized
            fake = self.model.domains[destination].denormalize(output.fake, pair.target_ids)
            losses[f"{route}/root_velocity"] = cfg.root_velocity_lambda * F.mse_loss(
                mapped_root_velocity(original[..., -4:-1], src_ref.min_velocity, src_ref.max_velocity,
                                     zero_safe=src_ref.protocol == "human_lafan_style_v1"),
                mapped_root_velocity(fake[..., -4:-1], dst_ref.min_velocity, dst_ref.max_velocity,
                                     zero_safe=dst_ref.protocol == "human_lafan_style_v1"),
            )
            losses[f"{route}/yaw"] = cfg.root_velocity_lambda * F.mse_loss(original[..., -1:], fake[..., -1:])
        losses.update({name: cfg.adversarial_lambda * loss for name, loss in self.compute_adversarial_losses(state).items()})
        return losses

    def compute_adversarial_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                domain = self.model.domains[destination]
                scores = domain.discriminator(domain.discriminator_input(pair.fake_bones))
                losses[f"{source}_to_{destination}/adversarial"] = generator_adversarial_loss(scores)
        return losses

    def compute_discriminator_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                domain = self.model.domains[destination]
                fake = domain.discriminator_input(pair.fake_bones).detach()
                if not self.model.is_reference:
                    fake = self.buffers[destination].query(fake)
                losses[f"{destination}/discriminator"] = discriminator_loss(
                    domain.discriminator(domain.discriminator_input(state.sources[destination].bones)),
                    domain.discriminator(fake),
                )
        return losses

    @torch.no_grad()
    def visualize(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        save_dir: str | Path,
        fps: float = 30.0,
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        batch = {domain: {key: value[:1] for key, value in item.items()} for domain, item in batch.items()}
        state = self.forward_state(batch)
        domains = self.model.domains

        for (source, destination), pair in state.pairs.items():
            if source == destination:
                continue

            scale = .01 if self.model.is_reference else 1
            positions, parents = side_by_side(
                domains[source].fk.world_positions(state.sources[source].bones)[0] * scale,
                torch.tensor(domains[source].fk.parents),
                domains[destination].fk.world_positions(pair.fake_bones)[0] * scale,
                torch.tensor(domains[destination].fk.parents),
            )
            visualize_motion(
                positions,
                parent_indices=parents,
                fps=fps,
                save_path=save_dir / f"{source}_to_{destination}.gif",
            )

    def checkpoint_metadata(self) -> Dict[str, Any]:
        if self.model.is_reference:
            regenerated = any(spec.reference.statistics_source != "archive" for spec in self.domain_specs.values())
            human_style = all(spec.reference.protocol == "human_lafan_style_v1" for spec in self.domain_specs.values())
            return {
                "representation": ("cmu_amass_lafan_style_v1" if human_style else
                                   "lafan1_dog_reference_raw_v1" if regenerated else "lafan1_dog_reference_archive_v1"),
                "root_frame": "heading_removed_local_velocity_inclusive_sum",
                "physical_units": "centimeters",
                "normalization_scope": ("reference_train_and_test_domain_statistics_with_raw_regeneration" if regenerated
                                        else "saved_reference_train_and_test_domain_statistics"),
                "routes": "cross_only_destination_batch_skeletons",
                "discriminator": ("raw_scores_all_integrated_available_positions" if human_style
                                  else "raw_scores_all_integrated_positions_with_end_sites"),
                "domains": {
                    name: {"kind": spec.reference.kind, "parents": spec.parents.tolist(),
                           "parents_withend": spec.reference.parents_withend.tolist(),
                           "not_end": spec.reference.not_end.tolist(), "body_parts": spec.parts,
                           "num_skeleton_banks": spec.num_banks,
                           "min_velocity": spec.reference.min_velocity, "max_velocity": spec.reference.max_velocity,
                           "statistics_sha256": spec.reference.statistics_sha256,
                           **({"protocol": spec.reference.protocol,
                               "selected_positions": spec.reference.position_indices,
                               "heading": "rest_anatomy_cross_up_right_project_xz_atan2",
                               "zero_speed": "exact_zero_maps_to_zero"} if human_style else {}),
                           "skeleton_banks_sha256": hashlib.sha256(
                               spec.offsets.cpu().numpy().tobytes() + spec.reference.offsets_withend.cpu().numpy().tobytes()
                           ).hexdigest()}
                    for name, spec in self.domain_specs.items()
                },
            }
        return {
            "representation": "joint_local_quaternions_root_velocity_xyz_yaw",
            "root_frame": "world_heading_kept_exclusive_velocity_sum",
            "mask_version": "joint_parts_shared_root_and_velocity_v1",
            "augmentation": "none",
            "physical_units": "meters",
            "normalization_scope": "per_skeleton_bank_all_channels_before_holdout",
            "domains": {
                name: {
                    "parents": spec.parents.tolist(),
                    "body_parts": spec.parts,
                    "num_skeleton_banks": spec.num_banks,
                    "end_effector_ids": spec.ee_ids,
                    # Older generic checkpoints claimed metres even when their
                    # AMASS/CMU banks were in cm. Do not silently restore those.
                    "skeleton_banks_sha256": hashlib.sha256(spec.offsets.cpu().numpy().tobytes()).hexdigest(),
                    "statistics_sha256": hashlib.sha256(
                        spec.mean.cpu().numpy().tobytes() + spec.std.cpu().numpy().tobytes()
                    ).hexdigest(),
                }
                for name, spec in self.domain_specs.items()
            },
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "metadata": self.checkpoint_metadata(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "metadata" not in state_dict:
            raise ValueError(
                "This checkpoint has no PAN metadata, so it predates this PAN implementation and cannot be loaded."
            )

        expected, found = self.checkpoint_metadata(), state_dict["metadata"]
        different = [key for key in expected if found.get(key) != expected[key]]
        if different:
            raise ValueError(f"Checkpoint is incompatible with this PAN setup, differing in: {different}")

        self.model.load_state_dict(state_dict["model"])
        self.generator_optimizer.load_state_dict(state_dict["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])

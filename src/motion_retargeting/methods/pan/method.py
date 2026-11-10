from __future__ import annotations

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
from .losses import discriminator_loss, end_effector_terms, generator_adversarial_loss
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
            for domain in self.domain_specs
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

    def compute_adversarial_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                scores = self.model.domains[destination].discriminator(pair.fake_bones)
                losses[f"{source}_to_{destination}/adversarial"] = generator_adversarial_loss(scores)
        return losses

    def compute_discriminator_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                discriminator = self.model.domains[destination].discriminator
                fake = self.buffers[destination].query(pair.fake_bones.detach())
                losses[f"{destination}/discriminator"] = discriminator_loss(
                    discriminator(state.sources[destination].bones),
                    discriminator(fake),
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

            positions, parents = side_by_side(
                domains[source].fk.world_positions(state.sources[source].bones)[0],
                self.domain_specs[source].parents,
                domains[destination].fk.world_positions(pair.fake_bones)[0],
                self.domain_specs[destination].parents,
            )
            visualize_motion(
                positions,
                parent_indices=parents,
                fps=fps,
                save_path=save_dir / f"{source}_to_{destination}.gif",
            )

    def checkpoint_metadata(self) -> Dict[str, Any]:
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

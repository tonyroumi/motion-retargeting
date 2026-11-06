"""
SAN (Skeleton-Aware Networks, Aberman et al. 2020) as a retargeting method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from motion_retargeting.methods.base import RetargetingMethod
from motion_retargeting.resources.replay_buffer import ReplayBuffer

from .config import DomainSpec, SANConfig, SANModelConfig
from .kinematics import end_effector_velocity
from .losses import adversarial_loss, end_effector_terms, generator_adversarial_loss
from .model import SANModel
from .utils import (
    ForwardState,
    PairState,
    SourceState,
    mpjpe,
    require_finite,
    side_by_side,
    squared_error_over_height2,
    total,
)

# The official commit this computation was ported from; the source of truth where it disagrees with the paper.
OFFICIAL_COMMIT = "74fff8fb35e6378351d03fb14ee22fccae28b0bf"


class SANMethod(RetargetingMethod):
    """
    SAN retargeting method. Owns the per-domain networks (`SANModel`) and the
    generator / discriminator optimizers. Computes what the official code does: edge rotations and
    absolute root in, bone-space discriminators, and the official objective and update order.
    """

    def __init__(
        self,
        model_cfg: SANModelConfig,
        domain_specs: Dict[str, DomainSpec],
        learning_rate: float = 2e-4,
        device: str | torch.device = "cpu",
        cfg: SANConfig | None = None,
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.cfg = cfg if cfg is not None else SANConfig()
        self.domain_specs = domain_specs

        self.device = torch.device(device)

        self.model = SANModel(model_cfg, domain_specs).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.model.generator_parameters(), lr=learning_rate, betas=tuple(self.cfg.betas)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator_parameters(), lr=learning_rate, betas=tuple(self.cfg.betas)
        )

        self.buffers = {
            domain: ReplayBuffer(
                self.cfg.replay_buffer_size,
                self.cfg.swap_probability,
            )
            for domain in self.domain_specs
        }

        # Counts train steps, for `cfg.loss_grad_norm_interval`.
        self.num_steps = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        One generator forward, then in order:
          - generator (encoders + decoders) update, with discriminator parameters frozen; its
            adversarial gradient still flows through them and FK
          - discriminator update, on the fakes of that same forward (or their pool replacements)
        Raises before stepping either optimizer if a route has no generated contact, or if the
        generator's objective is not finite.
        Returns scalar metrics for logging, keyed "<domain or route>/<term>" (plus "total/...").
        """
        state = self.forward_state(batch)

        interval = self.cfg.loss_grad_norm_interval
        log_loss_grads = interval > 0 and self.num_steps % interval == 0
        self.num_steps += 1

        self.set_discriminator_requires_grad(False)
        generator_losses = self.compute_generator_losses(state)
        require_finite(generator_losses)
        generator_loss, generator_info = total(generator_losses, "total/generator")
        generator_info.update(self.contact_fractions(state))
        if log_loss_grads:
            generator_info.update(
                self.loss_grad_norms(generator_losses, list(self.model.generator_parameters()))
            )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        grad_info = self.grad_norms(("motion_encoder", "offset_encoder", "decoder"))
        self.generator_optimizer.step()

        self.set_discriminator_requires_grad(True)
        discriminator_losses = self.compute_discriminator_losses(state)
        require_finite(discriminator_losses)
        discriminator_loss, discriminator_info = total(discriminator_losses, "total/discriminator")
        if log_loss_grads:
            discriminator_info.update(
                self.loss_grad_norms(discriminator_losses, list(self.model.discriminator_parameters()))
            )
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        grad_info.update(self.grad_norms(("discriminator",)))
        self.discriminator_optimizer.step()

        return {**generator_info, **discriminator_info, **grad_info}

    def grad_norms(self, components: tuple[str, ...]) -> Dict[str, float]:
        """
        L2 norm of the current parameter gradients of each (domain, component), keyed
        "grad/<domain>/<component>". Parameters without a gradient are skipped; a component
        with none at all reports 0.
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
        "loss_grad/<term key>" (e.g. "loss_grad/group_a/end_effector"). Costs one extra
        backward pass per term; the graph is retained for the step's own backward.
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
        """
        Unpaired diagnostics, keyed "<domain>/val/<metric>" for source motion of <domain>:
          - reconstruction_mse:                        the normalized network channels
          - reconstruction_mpjpe_m:                    Euclidean world joint error, in metres
          - reconstruction_squared_error_over_height2: mean squared world coordinate error / height^2, the
                                                       form of the official metric on an unpaired task
          - retarget_latent_l1, retarget_realism, retarget_ee_velocity: of the cross-domain route

        There is no paired ground truth here: none of these is paired retarget accuracy, and none is
        comparable to a paper score. The holdout's statistics include its own windows.
        """
        state = self.forward_state(batch)
        domains = self.model.domains
        metrics = {}

        for name, source in state.sources.items():
            other = next(domain for domain in domains if domain != name)
            pair = state.pairs[(name, other)]
            fk = domains[name].fk

            positions = fk.world_positions(source.bones)
            reconstruction = fk.world_positions(source.reconstruction_bones)

            prefix = f"{name}/val/"
            metrics.update({
                prefix + "reconstruction_mse": F.mse_loss(source.reconstruction, source.motion).item(),
                prefix + "reconstruction_mpjpe_m": mpjpe(reconstruction, positions),
                prefix + "reconstruction_squared_error_over_height2": squared_error_over_height2(
                    reconstruction, positions, domains[name].heights[source.bank_ids]
                ),
                prefix + "retarget_latent_l1": F.l1_loss(
                    state.outputs.pairs[(name, other)].fake_code, state.outputs.motion_codes[name]
                ).item(),
                prefix + "retarget_realism": domains[other].discriminate(pair.fake_bones).mean().item(),
                prefix + "retarget_ee_velocity": F.mse_loss(
                    pair.fake_velocity,
                    self.normalized_ee_velocity(name, source.bones, source.bank_ids),
                ).item(),
            })

        return metrics

    def set_discriminator_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.model.discriminator_parameters():
            parameter.requires_grad_(requires_grad)

    def forward_state(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> ForwardState:
        """ One generator forward, with each route's FK on the bank that route's statistics belong to. """
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
            fake_bones = domains[destination].bones(pair.fake, pair.target_ids)
            pairs[(source, destination)] = PairState(
                target_ids=pair.target_ids,
                fake_bones=fake_bones,
                fake_velocity=self.normalized_ee_velocity(destination, fake_bones, pair.target_ids),
            )

        return ForwardState(outputs=outputs, sources=sources, pairs=pairs)

    def normalized_positions(self, domain: str, bones: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Root-relative positions with the absolute root entry, over each sample's height. """
        modules = self.model.domains[domain]
        return modules.fk.root_relative_with_absolute_root(bones) / modules.heights[bank_ids][:, None, None, None]

    def normalized_ee_velocity(self, domain: str, bones: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Scaled world end-effector displacement per frame, over each sample's height. """
        modules = self.model.domains[domain]
        velocity = end_effector_velocity(modules.fk.world_positions(bones), modules.end_effectors)
        return velocity / modules.heights[bank_ids][:, None, None, None]

    def contact_fractions(self, state: ForwardState) -> Dict[str, float]:
        """
        Share of generated end-effector samples below the contact threshold, keyed
        "<source>_to_<destination>/contact_fraction". A route reaching zero stops training, so this is
        the early warning.
        """
        return {
            f"{source}_to_{destination}/contact_fraction": (
                torch.linalg.vector_norm(pair.fake_velocity, dim=-1) < self.cfg.contact_velocity_threshold
            ).float().mean().item()
            for (source, destination), pair in state.pairs.items()
        }

    def compute_generator_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        """
        Weighted generator terms, each logged on its own:
          - "<domain>/reconstruction": every normalized edge and root channel
          - "<domain>/root_position":  denormalized root XYZ over the source's height
          - "<domain>/joint_position": root-relative positions over the source's height
        and on all four "<source>_to_<destination>" routes:
          - latent_consistency:     L1 between the source's code and the fake's re-encoded code
          - end_effector / contact: velocity transfer, each side over its own skeleton's height
        plus adversarial on the two cross-domain routes.
        """
        cfg = self.cfg
        domains = self.model.domains
        losses = {}

        position_lambda = cfg.reconstruction_lambda * cfg.position_lambda

        velocities = {}
        for name, source in state.sources.items():
            heights = domains[name].heights[source.bank_ids][:, None, None]
            velocities[name] = self.normalized_ee_velocity(name, source.bones, source.bank_ids)

            losses[f"{name}/reconstruction"] = cfg.reconstruction_lambda * F.mse_loss(source.reconstruction, source.motion)
            losses[f"{name}/root_position"] = position_lambda * cfg.root_position_lambda * F.mse_loss(
                source.reconstruction_denormalized[..., -3:] / heights,
                source.denormalized[..., -3:] / heights,
            )
            losses[f"{name}/joint_position"] = position_lambda * F.mse_loss(
                self.normalized_positions(name, source.reconstruction_bones, source.bank_ids),
                self.normalized_positions(name, source.bones, source.bank_ids),
            )

        for (source, destination), pair in state.pairs.items():
            route = f"{source}_to_{destination}"

            losses[f"{route}/latent_consistency"] = cfg.cycle_consistency_lambda * F.l1_loss(
                state.outputs.motion_codes[source],
                state.outputs.pairs[(source, destination)].fake_code,
            )

            transfer, contact = end_effector_terms(velocities[source], pair.fake_velocity, cfg.contact_velocity_threshold)
            losses[f"{route}/end_effector"] = cfg.end_effector_lambda * transfer
            losses[f"{route}/contact"] = cfg.end_effector_lambda * cfg.contact_lambda * contact

            if source != destination:
                losses[f"{route}/adversarial"] = cfg.adversarial_lambda * generator_adversarial_loss(
                    domains[destination].discriminate(pair.fake_bones)
                )

        return losses

    def compute_discriminator_losses(self, state: ForwardState) -> Dict[str, torch.Tensor]:
        """
        Least-squares discriminator loss per domain (see `adversarial_loss`, logged as
        "<domain>/discriminator") over bone-space positions: the domain's real bones are pushed to 1,
        motion retargeted into the domain to 0. Fakes are detached and mixed with past ones through
        the domain's replay buffer.
        """
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                discriminate = self.model.domains[destination].discriminate
                fake = self.buffers[destination].query(pair.fake_bones.detach())
                losses[f"{destination}/discriminator"] = adversarial_loss(
                    discriminate(state.sources[destination].bones),
                    discriminate(fake),
                )
        return losses

    @torch.no_grad()
    def visualize(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        save_dir: str | Path,
        fps: float = 30.0,
    ) -> None:
        """
        Save one animation per cross-domain route of the batch's first sample: the source motion (left)
        next to it on a drawn target bank (right), both absolute-root world positions with no external
        heading restored, as "<source>_to_<destination>.gif" in `save_dir`.
        """
        from motion_retargeting.data.visualization.visualize import visualize_motion

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        batch = {domain: {key: value[:1] for key, value in item.items()} for domain, item in batch.items()}
        state = self.forward_state(batch)
        domains = self.model.domains

        for (source, destination), pair in state.pairs.items():
            if source == destination:
                continue

            merged, parents = side_by_side(
                domains[source].fk.world_positions(state.sources[source].bones)[0],
                self.domain_specs[source].parents,
                domains[destination].fk.world_positions(pair.fake_bones)[0],
                self.domain_specs[destination].parents,
            )
            visualize_motion(
                merged,
                parent_indices=parents,
                fps=fps,
                save_path=save_dir / f"{source}_to_{destination}.gif",
            )

    def computation_metadata(self) -> Dict[str, Any]:
        """ What a checkpoint's weights mean. The banks' values live in the model's persistent buffers. """
        return {
            "official_commit": OFFICIAL_COMMIT,
            "representation": "edge_parent_quaternions_absolute_root_xyz",
            "physical_units": "metres_with_static_path_offsets_in_centimetres",
            "normalization_scope": "per_skeleton_bank_all_channels_before_holdout",
            "kept_accidents": [
                "global_token_reads_edge_0_neighbors",
                "joint_offset_rows_through_edge_masks",
                "contacts_selected_on_generated_velocity",
                "discriminator_masks_and_pools_trained",
            ],
            "domains": {
                name: {
                    "parents": spec.parents.tolist(),
                    "num_skeleton_banks": spec.num_banks,
                    "end_effector_ids": spec.end_effectors,
                }
                for name, spec in self.domain_specs.items()
            },
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "computation": self.computation_metadata(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Rejects checkpoints of any other computation or setup rather than reusing their weights. """
        if "computation" not in state_dict:
            raise ValueError(
                "This checkpoint has no computation metadata, so it predates SAN's port of the official "
                "code and its weights do not fit these networks."
            )

        expected, found = self.computation_metadata(), state_dict["computation"]
        different = [key for key in expected if found.get(key) != expected[key]]
        if different:
            raise ValueError(f"Checkpoint is incompatible with this SAN setup, differing in: {different}")

        self.model.load_state_dict(state_dict["model"])
        self.generator_optimizer.load_state_dict(state_dict["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])

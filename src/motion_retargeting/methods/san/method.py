"""
SAN (Skeleton-Aware Networks, Aberman et al. 2020) as a retargeting method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from motion_retargeting.kinematics import ForwardKinematics
from motion_retargeting.methods.base import RetargetingMethod
from motion_retargeting.resources.replay_buffer import ReplayBuffer

from .config import DomainSpec, SANConfig, SANModelConfig
from .losses import adversarial_loss, end_effector_loss, generator_adversarial_loss
from .model import SANModel
from .utils import (
    DomainPositions,
    GeneratorOutputs,
    denormalize,
    end_effector_velocity,
    mpjpe,
    restore_motion,
    side_by_side,
    skeleton_height,
    split_by_domain,
    total,
)


class SANMethod(RetargetingMethod):
    """
    SAN retargeting method. Owns the per-domain networks (`SANModel`) and the
    generator / discriminator optimizers.
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
        One full training step over a batch from every domain:
          - generator (encoders + decoders) update
          - discriminator update
        Returns scalar metrics for logging, keyed "<domain>/<term>" (plus "total/...").
        """
        outputs = self.model.forward_generator(batch)
        positions = self.compute_positions(batch, outputs)

        interval = self.cfg.loss_grad_norm_interval
        log_loss_grads = interval > 0 and self.num_steps % interval == 0
        self.num_steps += 1

        # Generator update. Its adversarial terms also leave gradients on the discriminators;
        # those are cleared by the discriminator's zero_grad below.
        generator_losses = self.compute_generator_losses(batch, outputs, positions)
        generator_loss, generator_info = total(generator_losses, "total/generator")
        if log_loss_grads:
            generator_info.update(
                self.loss_grad_norms(generator_losses, list(self.model.generator_parameters()))
            )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        grad_info = self.grad_norms(("motion_encoder", "offset_encoder", "decoder"))
        self.generator_optimizer.step()

        # Discriminator update, on detached (replayed) fakes.
        discriminator_losses = self.compute_discriminator_losses(positions)
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
        Validation metrics, keyed "<domain>/val/<metric>" where <domain> is the domain the
        *source* motion comes from and "other" is the remaining domain. `*_mse` is measured
        in the normalized network features, `*mpjpe` in joint positions (meters), and
        `retarget_ee_velocity` in skeleton heights per frame.
        """
        outputs = self.model.forward_generator(batch)
        by_domain = split_by_domain(outputs)
        positions = self.compute_positions(batch, outputs)
        metrics = {}

        for domain, out in by_domain.items():
            other = next(d for d in by_domain if d != domain)
            motion = batch[domain]["motion"]

            realism = self.model.domains[other].discriminate(positions[other].retargeted_into).mean()

            prefix = f"{domain}/val/"
            metrics.update({
                prefix + "retarget_latent": F.l1_loss(out.retargeted_code, out.motion_code).item(),
                prefix + "retarget_realism": realism.item(),
                prefix + "retarget_ee_velocity": F.mse_loss(
                    self.normalized_ee_velocity(positions[other].retargeted_into, batch[other]["offsets"], other),
                    self.normalized_ee_velocity(positions[domain].real, batch[domain]["offsets"], domain),
                ).item(),
                prefix + "reconstruction_mse": F.mse_loss(out.reconstruction, motion).item(),
                prefix + "reconstruction_mpjpe": mpjpe(positions[domain].reconstruction, positions[domain].real),
            })

        return metrics

    def compute_positions(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        outputs: GeneratorOutputs,
    ) -> Dict[str, DomainPositions]:
        """
        Global joint positions on each domain's skeleton. Motion retargeted into a domain
        keeps the heading of its source window.
        """
        by_domain = split_by_domain(outputs)
        positions = {}

        for domain, out in by_domain.items():
            other = next(d for d in by_domain if d != domain)
            motion, offsets, q_heading = batch[domain]["motion"], batch[domain]["offsets"], batch[domain]["q_heading"]
            spec = self.domain_specs[domain]

            positions[domain] = DomainPositions(
                real=self.forward_kinematics(motion, q_heading, offsets, spec),
                reconstruction=self.forward_kinematics(out.reconstruction, q_heading, offsets, spec),
                retargeted_into=self.forward_kinematics(out.retargeted_into, batch[other]["q_heading"], offsets, spec),
            )

        return positions

    def normalized_ee_velocity(self, positions: torch.Tensor, offsets: torch.Tensor, domain: str) -> torch.Tensor:
        """ End-effector velocities of `domain`'s skeleton, in skeleton heights per frame. """
        spec = self.domain_specs[domain]
        height = skeleton_height(offsets, spec.parents)
        return end_effector_velocity(positions, spec.end_effectors) / height[:, None, None, None]

    def compute_generator_losses(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        outputs: GeneratorOutputs,
        positions: Dict[str, DomainPositions],
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted generator losses, one set per domain (logged as "<domain>/<term>"). Positions
        and end-effector velocities are normalized by the height of their skeleton:
          - reconstruction:     d -> d matches the input
          - root_position:      d -> d matches the input's root positions
          - joint_position:     d -> d matches the input's joint positions
          - latent_consistency: d's motion code matches the code of d -> d and of d -> other, re-encoded
          - end_effector:       d -> d and d -> other match the input's end-effector velocities
          - adversarial:        other -> d looks real (score 1) to d's discriminator
        """
        cfg = self.cfg
        losses = {}

        for domain, out in split_by_domain(outputs).items():
            other = next(d for d in batch if d != domain)
            motion, offsets = batch[domain]["motion"], batch[domain]["offsets"]
            pos, other_pos = positions[domain], positions[other]

            height = skeleton_height(offsets, self.domain_specs[domain].parents)[:, None, None, None]
            real, reconstruction = pos.real / height, pos.reconstruction / height

            reconstruction_lambda = cfg.reconstruction_lambda
            position_lambda = reconstruction_lambda * cfg.position_lambda

            losses[f"{domain}/reconstruction"] = reconstruction_lambda * F.mse_loss(out.reconstruction, motion)
            losses[f"{domain}/root_position"] = position_lambda * cfg.root_position_lambda * F.mse_loss(
                reconstruction[:, :, 0], real[:, :, 0]
            )
            losses[f"{domain}/joint_position"] = position_lambda * F.mse_loss(reconstruction, real)

            losses[f"{domain}/latent_consistency"] = cfg.cycle_consistency_lambda * (
                F.l1_loss(out.motion_code, out.reconstructed_code)
                + F.l1_loss(out.motion_code, out.retargeted_code)
            )

            target_ee = self.normalized_ee_velocity(pos.real, offsets, domain)
            losses[f"{domain}/end_effector"] = cfg.end_effector_lambda * sum(
                end_effector_loss(predicted, target_ee, cfg.contact_lambda, cfg.contact_velocity_threshold)
                for predicted in (
                    self.normalized_ee_velocity(pos.reconstruction, offsets, domain),
                    self.normalized_ee_velocity(other_pos.retargeted_into, batch[other]["offsets"], other),
                )
            )

            losses[f"{domain}/adversarial"] = cfg.adversarial_lambda * generator_adversarial_loss(
                self.model.domains[domain].discriminate(pos.retargeted_into)
            )

        return losses

    def compute_discriminator_losses(self, positions: Dict[str, DomainPositions]) -> Dict[str, torch.Tensor]:
        """
        Least-squares discriminator loss per domain (see `adversarial_loss`, logged as
        "<domain>/discriminator") over global joint positions: the domain's real motion is
        pushed to 1, motion retargeted into the domain to 0. Fakes are detached and mixed
        with past ones through the domain's replay buffer.
        """
        losses = {}

        for domain, pos in positions.items():
            discriminate = self.model.domains[domain].discriminate

            fake = self.buffers[domain].query(pos.retargeted_into.detach())

            fake_scores = discriminate(fake)
            real_scores = discriminate(pos.real)

            losses[f"{domain}/discriminator"] = adversarial_loss(real_scores, fake_scores)

        return losses

    @torch.no_grad()
    def visualize(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        save_dir: str | Path,
        fps: float = 30.0,
    ) -> None:
        """
        Save one animation per source domain of the batch's first sample being retargeted:
        the source motion (left) next to it retargeted onto the other domain's skeleton
        (right), as "<domain>_to_<other>.gif" in `save_dir`.
        """
        from motion_retargeting.data.visualization.visualize import visualize_motion

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        batch = {domain: {key: value[:1] for key, value in item.items()} for domain, item in batch.items()}
        positions = self.compute_positions(batch, self.model.forward_generator(batch))

        for domain in positions:
            other = next(d for d in positions if d != domain)
            spec, other_spec = self.domain_specs[domain], self.domain_specs[other]

            source = positions[domain].real[0]
            retargeted = positions[other].retargeted_into[0]  # domain -> other

            merged, parents = side_by_side(source, spec.parents, retargeted, other_spec.parents)
            visualize_motion(
                merged,
                parent_indices=parents,
                fps=fps,
                save_path=save_dir / f"{domain}_to_{other}.gif",
            )

    @staticmethod
    def forward_kinematics(
        motion: torch.Tensor,
        q_heading: torch.Tensor,
        offsets: torch.Tensor,
        spec: DomainSpec,
    ) -> torch.Tensor:
        """
        Global joint positions for a batch of (normalized) motion windows of one domain.
        """
        denormalized_motion = denormalize(motion.clone(), spec)

        local_rotations = denormalized_motion[..., :-1, :]  # [B, W, J, 4]
        velocity = denormalized_motion[..., -1, :3]         # [B, W, 3], root displacement from frame t to t+1

        local_rotations, velocity = restore_motion(local_rotations, velocity, q_heading)

        # Frame t's root position is the sum of the displacements of frames 0..t-1.
        root_positions = torch.cumsum(velocity, dim=1) - velocity

        # Per-sample offsets, broadcast over the window's frames.
        offsets = offsets[:, None].expand(-1, motion.shape[1], -1, -1)  # [B, W, J, 3]

        fk = ForwardKinematics(spec.parents, offsets)
        positions, _ = fk(local_rotations, root_positions)
        return positions

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.generator_optimizer.load_state_dict(state_dict["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])

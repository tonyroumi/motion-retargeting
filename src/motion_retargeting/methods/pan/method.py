from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from motion_retargeting.kinematics import ForwardKinematics
from motion_retargeting.methods.base import RetargetingMethod
from motion_retargeting.resources.replay_buffer import ReplayBuffer
from motion_retargeting.data.visualization.visualize import visualize_motion

from .config import DomainSpec, PANConfig, PANModelConfig
from .losses import adversarial_loss
from .model import PANModel, GeneratorOutputs
from .utils import GeneratorOutputs, restore_motion, split_by_domain, total, side_by_side, mpjpe, denormalize


class PANMethod(RetargetingMethod):
    """
    PAN retargeting method. Owns the per-domain networks (`PANModel`) and the
    generator / discriminator optimizers.
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
        self.cfg = cfg
        self.domain_specs = domain_specs

        self.device = torch.device(device)

        self.model = PANModel(model_cfg, domain_specs).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.model.generator_parameters(), lr=learning_rate
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator_parameters(), lr=learning_rate
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
          - discriminator update
          - generator (encoders + decoders) update
        Returns scalar metrics for logging, keyed "<domain>/<term>" (plus "total/...").
        """
        outputs = self.model.forward_generator(batch)

        interval = self.cfg.loss_grad_norm_interval
        log_loss_grads = interval > 0 and self.num_steps % interval == 0
        self.num_steps += 1

        # Discriminator update, on detached (replayed) fakes.
        discriminator_losses = self.compute_discriminator_losses(batch, outputs)
        discriminator_loss, discriminator_info = total(discriminator_losses, "total/discriminator")
        if log_loss_grads:
            discriminator_info.update(
                self.loss_grad_norms(discriminator_losses, list(self.model.discriminator_parameters()))
            )
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        # Read before the generator backward adds its adversarial gradients to the discriminators.
        grad_info = self.grad_norms(("discriminator",))
        if self.cfg.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.model.discriminator_parameters(), self.cfg.grad_norm_clip)
        self.discriminator_optimizer.step()

        # Generator update. Its adversarial terms also leave gradients on the discriminators;
        # those are cleared by the discriminator's zero_grad next step.
        generator_losses = self.compute_generator_losses(batch, outputs)
        generator_loss, generator_info = total(generator_losses, "total/generator")
        if log_loss_grads:
            generator_info.update(
                self.loss_grad_norms(generator_losses, list(self.model.generator_parameters()))
            )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        grad_info.update(self.grad_norms(("motion_encoder", "body_part_encoder", "decoder")))
        self.generator_optimizer.step()

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
        "loss_grad/<term key>" (e.g. "loss_grad/group_a/kinematic"). Costs one extra
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
        in the normalized network features, `*mpjpe` in joint positions (meters).
        """
        outputs = self.model.forward_generator(batch)
        by_domain = split_by_domain(outputs)
        metrics = {}

        for domain, out in by_domain.items():
            other = next(d for d in by_domain if d != domain)
            motion, offsets, q_heading = batch[domain]["motion"], batch[domain]["offsets"], batch[domain]["q_heading"]
            spec, _ = self.domain_specs[domain], self.domain_specs[other]

            reconstruction = out.reconstruction
            cycle = out.cycle

            realism = self.model.domains[other].discriminator(by_domain[other].retargeted_into).mean()

            positions = self.forward_kinematics(motion, q_heading, offsets, spec)

            prefix = f"{domain}/val/"
            metrics.update({
                prefix + "retarget_latent": F.mse_loss(out.retargeted_code, out.motion_code).item(),
                prefix + "retarget_realism": realism.item(),
                prefix + "reconstruction_mse": F.mse_loss(reconstruction, motion).item(),
                prefix + "reconstruction_mpjpe": mpjpe(self.forward_kinematics(reconstruction, q_heading, offsets, spec), positions),
                prefix + "cycle_mse": F.mse_loss(cycle, motion).item(),
                prefix + "cycle_mpjpe": mpjpe(self.forward_kinematics(cycle, q_heading, offsets, spec), positions),
            })

        return metrics

    def compute_generator_losses(self, batch: Dict[str, Dict[str, torch.Tensor]], outputs: GeneratorOutputs):
        """
        Weighted generator losses, one set per domain (logged as "<domain>/<term>"):
          - reconstruction:     d -> d matches the input
          - cycle_consistency:  d -> other -> d matches the input
          - latent_consistency: d's motion code matches the code of (d -> other) re-encoded by other
          - kinematic:          d -> d matches the input in joint-position space
          - adversarial:        other -> d looks real (logit 0) to d's discriminator
        """
        cfg = self.cfg
        losses = {}

        B, W = batch["group_a"]["motion"].shape[:2] 
        for domain, out in split_by_domain(outputs).items():
            motion, offsets, q_heading = batch[domain]["motion"], batch[domain]["offsets"], batch[domain]["q_heading"]
            spec = self.domain_specs[domain]
            discriminator = self.model.domains[domain].discriminator

            # check here
            positions = self.forward_kinematics(motion, q_heading, offsets, spec)
            reconstructed_positions = self.forward_kinematics(out.reconstruction, q_heading, offsets, spec)

            losses[f"{domain}/reconstruction"] = cfg.reconstruction_lambda * F.mse_loss(out.reconstruction, motion)
            losses[f"{domain}/cycle_consistency"] = cfg.cycle_consistency_lambda * F.mse_loss(out.cycle, motion)
            losses[f"{domain}/latent_consistency"] = cfg.cycle_consistency_lambda * F.mse_loss(
                out.retargeted_code, out.motion_code
            )
            losses[f"{domain}/kinematic"] = cfg.kinematic_lambda * F.mse_loss(reconstructed_positions, positions)
            losses[f"{domain}/adversarial"] = cfg.adversarial_lambda * torch.mean(
                discriminator(out.retargeted_into) ** 2
            )

        return losses

    def compute_discriminator_losses(self, batch: Dict[str, Dict[str, torch.Tensor]], outputs: GeneratorOutputs):
        """
        Least-squares discriminator loss per domain (see `adversarial_loss`, logged as
        "<domain>/discriminator"): the domain's real motion is pushed to 0, motion
        retargeted into the domain to 1. Fakes are detached and mixed with past ones
        through the domain's replay buffer.
        """
        losses = {}

        for domain, out in split_by_domain(outputs).items():
            motion = batch[domain]["motion"]
            if self.cfg.disc_grad_penalty:
                motion = motion.detach().requires_grad_(True)
            discriminator = self.model.domains[domain].discriminator

            fake = self.buffers[domain].query(out.retargeted_into.detach())

            real_logits = discriminator(motion)
            fake_logits = discriminator(fake)

            losses[f"{domain}/discriminator"] = adversarial_loss(real_logits, fake_logits)

            # L2 on the final (logit) layer's weights.
            if self.cfg.disc_logit_reg:
                logit_weights = torch.flatten(discriminator.conv3.weight)
                losses[f"{domain}/disc_logit_reg"] = self.cfg.disc_logit_reg * torch.sum(
                    torch.square(logit_weights)
                )

            # Zero-centered gradient penalty on real motion.
            if self.cfg.disc_grad_penalty:
                motion_gradient = torch.autograd.grad(
                    real_logits,
                    motion,
                    grad_outputs=torch.ones_like(real_logits),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                gradient_penalty = torch.sum(torch.square(motion_gradient.flatten(1)), dim=-1).mean()
                losses[f"{domain}/disc_grad_penalty"] = self.cfg.disc_grad_penalty * gradient_penalty

            # L2 on all conv weights.
            if self.cfg.disc_weight_decay:
                weights = [
                    torch.flatten(module.weight)
                    for module in discriminator.modules()
                    if isinstance(module, nn.Conv1d)
                ]
                weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                losses[f"{domain}/disc_weight_decay"] = self.cfg.disc_weight_decay * weight_decay

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
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        batch = {domain: {key: value[:1] for key, value in item.items()} for domain, item in batch.items()}
        by_domain = split_by_domain(self.model.forward_generator(batch))

        for domain in by_domain:
            other = next(d for d in by_domain if d != domain)
            spec, other_spec = self.domain_specs[domain], self.domain_specs[other]

            source = self.forward_kinematics(batch[domain]["motion"], batch[domain]["q_heading"], batch[domain]["offsets"], spec)[0]

            retargeted = self.forward_kinematics(
                by_domain[other].retargeted_into,  # domain -> other
                batch[domain]["q_heading"],
                batch[other]["offsets"],
                other_spec,
            )[0]

            positions, parents = side_by_side(source, spec.parents, retargeted, other_spec.parents)
            visualize_motion(
                positions,
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
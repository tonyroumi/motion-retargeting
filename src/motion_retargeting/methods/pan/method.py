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

from .config import (
    DomainSpec,
    PANConfig,
    PANModelConfig,
    PANReferenceConfig,
    PANReferenceDomainSpec,
    PANReferenceModelConfig,
    REFERENCE_RECIPE,
)
from .kinematics import end_effector_velocity
from .losses import (
    adversarial_loss,
    reference_discriminator_loss,
    reference_end_effector_terms,
    reference_generator_adversarial_loss,
)
from .masks import MASK_RECIPE
from .model import PANModel, PANReferenceModel, GeneratorOutputs
from .utils import (
    GeneratorOutputs,
    ReferenceForwardState,
    ReferencePairState,
    ReferenceSourceState,
    require_finite,
    restore_motion,
    split_by_domain,
    total,
    side_by_side,
    squared_error_over_height2,
    mpjpe,
    denormalize,
)

# The pinned reference commit the "mixamo_reference" recipe's computation was ported from.
REFERENCE_COMMIT = "5a51d929fbb3c804494f1d3c58e4558c3be8e35d"


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


class PANReferenceMethod(PANMethod):
    """
    PAN's reference-Mixamo recipe: edge rotations and absolute root in, masked networks, positional
    patch discriminators, and the reference objective and update order. Shares only the optimizer
    gradient diagnostics with the legacy adaptation.
    """

    def __init__(
        self,
        model_cfg: PANReferenceModelConfig,
        domain_specs: Dict[str, PANReferenceDomainSpec],
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
        cfg: PANReferenceConfig | None = None,
    ):
        RetargetingMethod.__init__(self)

        self.model_cfg = model_cfg
        self.cfg = cfg if cfg is not None else PANReferenceConfig()
        self.domain_specs = domain_specs

        self.device = torch.device(device)

        self.model = PANReferenceModel(model_cfg, domain_specs).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.model.generator_parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator_parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )

        self.buffers = {
            domain: ReplayBuffer(self.cfg.replay_buffer_size, self.cfg.swap_probability)
            for domain in self.domain_specs
        }

        self.num_steps = 0

    def train_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        One generator forward, then in order:
          - generator update, with discriminator parameters frozen; its adversarial gradient still
            flows through them and FK
          - discriminator update, on the fakes of that same forward (or their pool replacements)
        Raises before stepping either optimizer if the generator's objective is not finite.
        """
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

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Unpaired diagnostics, keyed "<domain>/val/<metric>" for source motion of <domain>:
          - *_mse:                          the normalized network channels
          - *_mpjpe_m:                      Euclidean world joint error, in metres
          - *_squared_error_over_height2:   mean squared world coordinate error / height^2, the form
                                            of the reference's metric on an unpaired, non-reference task
          - retarget_latent_l1, retarget_realism: of the cross-domain route

        There is no paired ground truth here: neither is paired retarget accuracy, and neither is
        comparable to a paper score. The holdout's statistics include its own windows.
        """
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

    def forward_state(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> ReferenceForwardState:
        """ One generator forward, with each route's FK on the bank that route's statistics belong to. """
        outputs = self.model.forward_generator(batch)
        domains = self.model.domains

        sources = {}
        for name, domain in domains.items():
            bank_ids, motion = batch[name]["bank_id"], batch[name]["motion"]
            reconstruction = outputs.reconstructions[name]
            sources[name] = ReferenceSourceState(
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
            pairs[(source, destination)] = ReferencePairState(
                target_ids=pair.target_ids,
                fake_bones=domains[destination].bones(pair.fake, pair.target_ids),
                cycle_bones=domains[source].bones(pair.cycle, sources[source].bank_ids),
            )

        return ReferenceForwardState(outputs=outputs, sources=sources, pairs=pairs)

    def normalized_positions(self, domain: str, bones: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Root-relative positions with the absolute root entry, over each sample's height. """
        modules = self.model.domains[domain]
        return modules.fk.root_relative_with_absolute_root(bones) / modules.heights[bank_ids][:, None, None, None]

    def normalized_end_effector_velocity(self, domain: str, bones: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Scaled world end-effector displacement per frame, over each sample's height. """
        modules = self.model.domains[domain]
        velocity = end_effector_velocity(modules.fk.world_positions(bones), modules.ee_ids)
        return velocity / modules.heights[bank_ids][:, None, None, None]

    def compute_generator_losses(self, state: ReferenceForwardState) -> Dict[str, torch.Tensor]:
        """
        Weighted generator terms, each logged on its own:
          - "<domain>/reconstruction": every normalized edge and root channel
          - "<domain>/root":           denormalized root XYZ over the source's height
          - "<domain>/position":       root-relative positions over the source's height
        and on all four "<source>_to_<destination>" routes:
          - latent_cycle:                L1 between the source's code and the fake's re-encoded code
          - position_cycle:              root-relative positions of the cycle against the source
          - end_effector / contact:      velocity transfer, each side over its own skeleton's height
        plus adversarial on the two cross-domain routes.
        """
        cfg = self.cfg
        losses = {}

        positions = {}
        velocities = {}
        for name, source in state.sources.items():
            heights = self.model.domains[name].heights[source.bank_ids][:, None, None]
            positions[name] = self.normalized_positions(name, source.bones, source.bank_ids)
            velocities[name] = self.normalized_end_effector_velocity(name, source.bones, source.bank_ids)

            losses[f"{name}/reconstruction"] = cfg.reconstruction_lambda * F.mse_loss(source.reconstruction, source.motion)
            losses[f"{name}/root"] = cfg.root_lambda * F.mse_loss(
                source.reconstruction_denormalized[..., -3:] / heights,
                source.denormalized[..., -3:] / heights,
            )
            losses[f"{name}/position"] = cfg.position_lambda * F.mse_loss(
                self.normalized_positions(name, source.reconstruction_bones, source.bank_ids),
                positions[name],
            )

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

            transfer, contact = reference_end_effector_terms(
                velocities[source],
                self.normalized_end_effector_velocity(destination, pair.fake_bones, pair.target_ids),
                cfg.contact_threshold,
            )
            losses[f"{route}/end_effector"] = cfg.end_effector_lambda * transfer
            losses[f"{route}/contact"] = cfg.end_effector_lambda * cfg.contact_lambda * contact

        adversarial = self.compute_adversarial_losses(state)
        losses.update({name: cfg.adversarial_lambda * loss for name, loss in adversarial.items()})

        return losses

    def compute_adversarial_losses(self, state: ReferenceForwardState) -> Dict[str, torch.Tensor]:
        """ Cross-domain routes only: each fake's positions, judged by its destination's discriminator. """
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                scores = self.model.domains[destination].discriminator(pair.fake_bones)
                losses[f"{source}_to_{destination}/adversarial"] = reference_generator_adversarial_loss(scores)
        return losses

    def compute_discriminator_losses(self, state: ReferenceForwardState) -> Dict[str, torch.Tensor]:
        """ Per domain: its real positions against fakes retargeted into it, detached and replayed. """
        losses = {}
        for (source, destination), pair in state.pairs.items():
            if source != destination:
                discriminator = self.model.domains[destination].discriminator
                fake = self.buffers[destination].query(pair.fake_bones.detach())
                losses[f"{destination}/discriminator"] = reference_discriminator_loss(
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
        """
        Save one animation per cross-domain route of the batch's first sample: the source motion (left)
        next to it on a drawn target bank (right), both absolute-root world positions with no external
        heading restored, as "<source>_to_<destination>.gif" in `save_dir`.
        """
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

    def recipe_metadata(self) -> Dict[str, Any]:
        """ What a checkpoint's weights mean. The banks' values live in the model's persistent buffers. """
        return {
            "recipe": REFERENCE_RECIPE,
            "reference_commit": REFERENCE_COMMIT,
            "representation": "edge_parent_quaternions_absolute_root_xyz",
            "physical_units": "metres",
            "normalization_scope": "per_skeleton_bank_all_channels_before_holdout",
            "mask_recipe": MASK_RECIPE,
            "domains": {
                name: {
                    "parents": spec.parents.tolist(),
                    "num_skeleton_banks": spec.num_banks,
                    "end_effector_ids": spec.ee_ids,
                }
                for name, spec in self.domain_specs.items()
            },
        }

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "recipe": self.recipe_metadata()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Rejects legacy or otherwise incompatible checkpoints rather than reusing their weights. """
        if "recipe" not in state_dict:
            raise ValueError(
                f"This checkpoint has no recipe metadata, so it is from the legacy PAN recipe and "
                f"cannot be loaded into '{REFERENCE_RECIPE}'."
            )

        expected, found = self.recipe_metadata(), state_dict["recipe"]
        different = [key for key in expected if found.get(key) != expected[key]]
        if different:
            raise ValueError(f"Checkpoint is incompatible with this '{REFERENCE_RECIPE}' setup, differing in: {different}")

        super().load_state_dict(state_dict)

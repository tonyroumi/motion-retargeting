from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from motion_retargeting.methods.base import RetargetingMethod
from motion_retargeting.resources.replay_buffer import ReplayBuffer
from motion_retargeting.data.visualization.visualize import visualize_motion

from .config import DomainSpec, PANConfig, PANModelConfig
from .data import batch_checksum
from .kinematics import end_effector_velocity
from .losses import discriminator_loss, end_effector_terms, generator_adversarial_loss
from .model import PANModel
from .utils import (
    ForwardState,
    NonFiniteGradientError,
    PairState,
    SourceState,
    require_finite,
    total,
    side_by_side,
    squared_error_over_height2,
    mpjpe,
)


GENERATOR_COMPONENTS = ("motion_encoder", "body_part_encoder", "decoder")
DISCRIMINATOR_COMPONENTS = ("discriminator",)
CAPTURE_FORMAT = 1
SPIKE_BASELINE_DECAY = 0.99


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
        self.spike_captures: List[Path] = []
        self.grad_baseline: float | None = None  # running mean of the generator's gradient norm

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Both updates of one step. Each optimizer steps only once both losses and both gradients are
        finite; on any error the step's pre-step state is captured (see `capture`) and the error re-raised.
        """
        context = self.replay_context()
        state, generator_losses, grad_info = None, {}, {}
        try:
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
            grad_info = self.grad_norms(GENERATOR_COMPONENTS)
            grad_info["grad/generator"] = self.global_norm(grad_info)
            self.require_finite_gradients(grad_info, GENERATOR_COMPONENTS)

            bound = self.spike_bound(grad_info["grad/generator"])
            if bound is not None:
                self.capture_spike(context, batch, state, generator_losses, grad_info, bound)

            if self.cfg.generator_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.generator_parameters(), self.cfg.generator_max_grad_norm)

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
            discriminator_grads = self.grad_norms(DISCRIMINATOR_COMPONENTS)
            grad_info.update(discriminator_grads)
            self.require_finite_gradients(discriminator_grads, DISCRIMINATOR_COMPONENTS)

            # Neither optimizer's step changes what the other's gradients were computed from, so
            # stepping both last is the generator-then-discriminator update, made only once both
            # gradients are known to be finite.
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()
        except Exception as error:
            self.capture_failure(context, batch, state, generator_losses, grad_info, error)
            raise

        return {**generator_info, **discriminator_info, **grad_info}

    def spike_bound(self, norm: float) -> float | None:
        """
        `grad_spike_factor` times the running baseline when `norm` exceeds it, else None. The baseline
        follows the earlier steps with each one capped at that bound, so a spike barely moves it but a
        lasting rise does.
        """
        factor = self.cfg.grad_spike_factor
        if factor <= 0:
            return None
        if self.grad_baseline is None:
            self.grad_baseline = norm
            return None
        bound = factor * self.grad_baseline
        self.grad_baseline = SPIKE_BASELINE_DECAY * self.grad_baseline + (1 - SPIKE_BASELINE_DECAY) * min(norm, bound)
        return bound if norm > bound else None

    @staticmethod
    def global_norm(norms: Dict[str, float]) -> float:
        """ L2 norm over per-module gradient norms, in float32 as clipping computes it. """
        return torch.linalg.vector_norm(torch.tensor(list(norms.values()))).item()

    def require_finite_gradients(self, norms: Dict[str, float], components: tuple[str, ...]) -> None:
        """ Stop before an optimizer step would apply a NaN or infinite gradient to the weights. """
        if all(math.isfinite(norm) for norm in norms.values()):
            return
        parameters = [
            f"{domain}/{component}/{name}"
            for domain, modules in self.model.domains.items()
            for component in components
            for name, parameter in getattr(modules, component).named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        bad = {name: norm for name, norm in norms.items() if not math.isfinite(norm)}
        raise NonFiniteGradientError(
            f"Non-finite gradient norms, no optimizer stepped on them: {bad}; parameters with NaN or "
            f"infinite gradient entries: {parameters or 'none, the finite entries overflow the norm'}",
            parameters,
        )

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

    def replay_context(self) -> Dict[str, Any]:
        """ Taken before a step runs: the step count, RNG states and replay buffers a capture needs. """
        return {
            "num_steps": self.num_steps,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                "python": random.getstate(),
            },
            # The buffers replace their items rather than modify them, so a shallow copy is a snapshot.
            "buffers": {
                name: {"items": list(buffer.items), "num_items": buffer.num_items}
                for name, buffer in self.buffers.items()
            },
        }

    def capture_spike(
        self,
        context: Dict[str, Any],
        batch: Dict[str, Dict[str, torch.Tensor]],
        state: ForwardState,
        losses: Dict[str, torch.Tensor],
        grad_info: Dict[str, float],
        bound: float,
    ) -> None:
        """ Capture and report a step with a large finite generator gradient; training continues. """
        try:
            report = self.step_report(context, batch, state, losses, grad_info)
            report["reason"] = (
                f"generator gradient norm {grad_info['grad/generator']:.4g} above {bound:.4g}, "
                f"{self.cfg.grad_spike_factor:g} times its running baseline"
            )
            path = self.capture(context, batch, "spike", report)
            print(self.format_report(report, path))
        except Exception as capture_error:
            print(f"[PAN] Could not capture the gradient spike: {capture_error!r}")

    def capture_failure(
        self,
        context: Dict[str, Any],
        batch: Dict[str, Dict[str, torch.Tensor]],
        state: ForwardState | None,
        losses: Dict[str, torch.Tensor],
        grad_info: Dict[str, float],
        error: Exception,
    ) -> None:
        """ Capture and report a failed step, never masking the error that failed it. """
        try:
            report = self.step_report(context, batch, state, losses, grad_info)
            report["reason"] = f"{type(error).__name__}: {error}"
            if isinstance(error, NonFiniteGradientError):
                report["nonfinite_gradient_parameters"] = error.parameters
            path = self.capture(context, batch, "failure", report)
            print(self.format_report(report, path))
        except Exception as capture_error:
            print(f"[PAN] Could not capture the failed step: {capture_error!r}")

    def capture(
        self,
        context: Dict[str, Any],
        batch: Dict[str, Dict[str, torch.Tensor]],
        kind: str,
        report: Dict[str, Any],
    ) -> Path | None:
        """
        Save <capture_dir>/step_<N>_<kind>.pt with what replaying the step needs: weights and optimizer
        states before either step, replay buffers, RNG states, the batch's provenance, config and
        `report`. Nothing is saved without a `capture_dir`; of the spike captures, only the latest
        `spike_capture_limit` are kept.
        """
        if self.capture_dir is None or (kind == "spike" and self.cfg.spike_capture_limit <= 0):
            return None

        path = Path(self.capture_dir) / f"step_{context['num_steps'] + 1:08d}_{kind}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CAPTURE_FORMAT,
                "kind": kind,
                "num_steps": context["num_steps"],
                "report": report,
                "method": self.state_dict(),
                "buffers": context["buffers"],
                "rng": context["rng"],
                "batch": self.batch_provenance(batch),
                "config": asdict(self.cfg),
            },
            path,
        )

        if kind == "spike":
            self.spike_captures.append(path)
            while len(self.spike_captures) > self.cfg.spike_capture_limit:
                self.spike_captures.pop(0).unlink(missing_ok=True)
        return path

    def batch_provenance(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Per domain: bank ids, dataset indices, reversal flags, windows and a checksum of the motion. The
        motion itself only with `capture_batch_motion`, or when the batch has no dataset indices.
        """
        provenance = {}
        for name, item in batch.items():
            entry = {
                key: item[key].detach().cpu()
                for key in ("bank_id", "index", "reversed", "window") if key in item
            }
            entry["checksum"] = batch_checksum(item["motion"])
            if self.cfg.capture_batch_motion or not self.has_provenance(item):
                entry["motion"] = item["motion"].detach().cpu()
            provenance[name] = entry
        return provenance

    @staticmethod
    def has_provenance(item: Dict[str, torch.Tensor]) -> bool:
        return "index" in item and "reversed" in item

    def restore_capture(self, capture: Dict[str, Any]) -> None:
        """
        A captured step's state before it ran: weights, optimizer states, replay buffers, step count
        and RNG. Its batch comes from `data.rebuild_batch`.
        """
        if capture.get("format") != CAPTURE_FORMAT:
            raise ValueError(f"Unsupported capture format {capture.get('format')}, expected {CAPTURE_FORMAT}")
        self.load_state_dict(capture["method"])
        for name, buffer in self.buffers.items():
            buffer.items = [item.to(self.device) for item in capture["buffers"][name]["items"]]
            buffer.num_items = capture["buffers"][name]["num_items"]
        self.num_steps = capture["num_steps"]

        rng = capture["rng"]
        # RNG states are CPU byte tensors, wherever the capture was loaded to.
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"].cpu(), self.device)
        random.setstate(rng["python"])

    @torch.no_grad()
    def step_report(
        self,
        context: Dict[str, Any],
        batch: Dict[str, Dict[str, torch.Tensor]],
        state: ForwardState | None,
        losses: Dict[str, torch.Tensor],
        grad_info: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        What identifies a step: its losses and gradient norms, per domain its largest normalized inputs
        (with their bank statistic) and longest root travel, each sample's target banks, and the
        weight and Adam state norms the step started from.
        """
        report = {
            "step": context["num_steps"] + 1,
            "losses": {name: loss.item() for name, loss in losses.items()},
            "grad_norms": dict(grad_info),
            "domains": {},
            "model": self.model_report(),
        }

        for name, item in batch.items():
            def identity(sample: int) -> Dict[str, Any]:
                bank = int(item["bank_id"][sample])
                found = {"sample": sample, "bank": bank}
                found.update({key: item[key][sample].tolist() for key in ("index", "reversed", "window") if key in item})
                if state is not None:
                    found["target_banks"] = {
                        destination: int(pair.target_ids[sample])
                        for (source, destination), pair in state.pairs.items() if source == name
                    }
                return found

            domain = self.model.domains[name]
            peak, channel = item["motion"].abs().amax(dim=1).max(dim=-1)  # [B]
            largest = []
            for sample in peak.argsort(descending=True)[:3].tolist():
                found = identity(sample)
                bank, index = found["bank"], int(channel[sample])
                found.update({
                    "max_abs_z": peak[sample].item(),
                    "channel": index,
                    "bank_mean": domain.mean[bank, index].item(),
                    "bank_std": domain.std[bank, index].item(),
                })
                largest.append(found)

            entry = {
                "provenance": "dataset" if self.has_provenance(item) else "unavailable, motion stored in the capture",
                "largest_inputs": largest,
            }
            if state is not None:
                travel = state.sources[name].bones[..., 0, :].norm(dim=-1).amax(dim=1)  # [B], metres from the start
                sample = int(travel.argmax())
                entry["longest_root_travel"] = {**identity(sample), "metres": travel[sample].item()}
            report["domains"][name] = entry

        return report

    @torch.no_grad()
    def model_report(self) -> Dict[str, Dict[str, float]]:
        """ Per (domain, component): weight norm, and its Adam state's largest second moment and step. """
        report = {}
        for optimizer, components in (
            (self.generator_optimizer, GENERATOR_COMPONENTS),
            (self.discriminator_optimizer, DISCRIMINATOR_COMPONENTS),
        ):
            for domain, modules in self.model.domains.items():
                for component in components:
                    parameters = list(getattr(modules, component).parameters())
                    states = [optimizer.state[p] for p in parameters if p in optimizer.state]
                    report[f"{domain}/{component}"] = {
                        "weight_norm": torch.linalg.vector_norm(torch.stack([p.norm() for p in parameters])).item(),
                        "adam_exp_avg_sq_max": max((s["exp_avg_sq"].max().item() for s in states), default=0.0),
                        "adam_step": float(states[0]["step"]) if states else 0.0,
                    }
        return report

    @staticmethod
    def format_report(report: Dict[str, Any], path: Path | None) -> str:
        """ A few lines for the console; the capture holds the full report. """
        grads = {name: norm for name, norm in report["grad_norms"].items() if name != "grad/generator"}
        top = sorted(grads.items(), key=lambda item: item[1] if math.isfinite(item[1]) else math.inf, reverse=True)[:3]
        lines = [f"[PAN] step {report['step']}: {report['reason']}"]
        if "grad/generator" in report["grad_norms"]:
            lines.append(
                f"    generator gradient norm {report['grad_norms']['grad/generator']:.4g}; largest: "
                + ", ".join(f"{name} {norm:.4g}" for name, norm in top)
            )
        else:
            lines.append("    no gradients computed yet")
        for name, entry in report["domains"].items():
            worst = entry["largest_inputs"][0]
            line = (
                f"    {name}: largest |z| {worst['max_abs_z']:.3g} at sample {worst['sample']} "
                f"(index {worst.get('index')}, window {worst.get('window')}, reversed {worst.get('reversed')}, "
                f"bank {worst['bank']}, channel {worst['channel']}, bank std {worst['bank_std']:.3g})"
            )
            if "longest_root_travel" in entry:
                travel = entry["longest_root_travel"]
                line += f"; longest root travel {travel['metres']:.3g} m at sample {travel['sample']}"
            if entry["provenance"] != "dataset":
                line += f"; provenance {entry['provenance']}"
            lines.append(line)
        lines.append(f"    capture: {path}" if path is not None else "    capture: not saved")
        return "\n".join(lines)

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
            losses[f"{name}/root"] = cfg.root_lambda * F.mse_loss(
                source.reconstruction_bones[..., 0, :] / heights,
                source.bones[..., 0, :] / heights,
            )
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

            transfer, contact = end_effector_terms(
                velocities[source],
                self.normalized_end_effector_velocity(destination, pair.fake_bones, pair.target_ids),
                cfg.contact_threshold,
            )
            losses[f"{route}/end_effector"] = cfg.end_effector_lambda * transfer
            losses[f"{route}/contact"] = cfg.end_effector_lambda * cfg.contact_lambda * contact

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
            "representation": "edge_parent_quaternions_root_velocity_xyz_yaw",
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

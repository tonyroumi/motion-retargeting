import logging
import os
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from .array import ArrayUtils


class Logger:
    _LOG_NAME = "moveitmoveit"
    _LOG_FILENAME = "run.log"
    _GROUP_ORDER = [
        "reward", "disc", "loss", "policy", "ratio",
        "log_prob", "advantage", "value", "grad", "policy_dim", "eval",
    ]
    _L, _R = 28, 13  # label and value column widths

    def __init__(
        self,
        log_dir: str = "logs",
        verbose: bool = False,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._log_path = os.path.join(log_dir, self._LOG_FILENAME)

        self._logger = logging.getLogger(self._LOG_NAME)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            )
            file_handler = logging.FileHandler(self._log_path, mode="a")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        self.writer = SummaryWriter(log_dir=log_dir)

        self.verbose = verbose

        self._metric_accum: dict[str, list[float]] = {}

        # Per-metric step counters. Incremented on every log_metric call so
        # callers never need to thread a global step through every call site.
        self._metric_steps: dict[str, int] = defaultdict(int)

        self.use_wandb = use_wandb
        self.wandb = None

        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=log_dir,
                )
            except ImportError as exc:
                raise RuntimeError("use_wandb=True but wandb is not installed") from exc

    def info(self, msg: str):
        self._logger.info(msg)

    def warn(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def log_metric(self, name: str, value: float, step_num: int | None = None) -> int:
        """Log a scalar metric.

        Parameters
        ----------
        name:
            Metric key, optionally namespaced with ``/`` (e.g. ``"loss/policy"``).
        value:
            Scalar value.  Tensors are detached and converted automatically.
        step_num:
            Global step to attach to TensorBoard / W&B.  When omitted the
            per-metric internal counter is used and returned so callers can
            reference it later.

        Returns
        -------
        int
            The step number actually used (useful when ``step_num`` is ``None``).
        """
        if ArrayUtils.is_tensor(value):
            value = value.detach().item()

        # Advance this metric's own counter regardless of whether an external
        # step was provided — keeps the clock ticking even in eval-only paths.
        self._metric_steps[name] += 1
        effective_step = step_num if step_num is not None else self._metric_steps[name]

        if self.verbose:
            self.info(f"{name}: {value:.6g} (step={effective_step})")

        self._metric_accum.setdefault(name, []).append(value)

        self.writer.add_scalar(name, value, effective_step)

        if self.use_wandb:
            self.wandb.log({name: value}, step=effective_step)

        return effective_step

    def get_step(self, name: str) -> int:
        """Return the current internal step counter for *name*."""
        return self._metric_steps[name]

    def reset_step(self, name: str) -> None:
        """Reset the internal step counter for *name* to zero."""
        self._metric_steps[name] = 0

    def log_metrics(self, metrics: dict[str, float], step_num: int | None = None) -> None:
        """Convenience wrapper — log an entire dict in one call."""
        for name, value in metrics.items():
            self.log_metric(name, value, step_num)

    def log_grads(
        self,
        model: nn.Module,
        *,
        step_num: int | None = None,
        log_per_param: bool = True,
        log_per_layer: bool = True,
        log_global: bool = True,
        prefix: str = "grad",
        norm_type: float = 2.0,
    ) -> dict[str, float]:
        """Log gradient norms for *model* after a backward pass.

        Three granularities are available and can be enabled independently:

        * **per-param** — one entry per ``named_parameters()`` leaf that has a
          gradient.  Key pattern: ``"{prefix}/param/{name}"``.
        * **per-layer** — groups parameters by their first name component
          (e.g. ``"encoder"``).  Key pattern: ``"{prefix}/layer/{layer}"``.
        * **global** — single scalar across every parameter.
          Key pattern: ``"{prefix}/global"``.

        Parameters
        ----------
        model:
            The network whose ``.grad`` buffers to inspect.
        step_num:
            Passed through to :meth:`log_metric`.
        log_per_param:
            Whether to emit a metric per parameter tensor.
        log_per_layer:
            Whether to emit a metric per top-level layer.
        log_global:
            Whether to emit a single global gradient norm.
        prefix:
            Namespace prefix for all gradient metric keys.
        norm_type:
            Which Lp norm to compute (default: L2).

        Returns
        -------
        dict[str, float]
            All gradient norm values that were logged, keyed by metric name.
        """
        layer_norms: dict[str, list[float]] = defaultdict(list)
        global_norms: list[float] = []
        logged: dict[str, float] = {}

        for param_name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.detach().norm(norm_type).item()
            global_norms.append(grad_norm ** norm_type)

            layer = param_name.split(".")[0]
            layer_norms[layer].append(grad_norm ** norm_type)

            if log_per_param:
                key = f"{prefix}/param/{param_name}"
                self.log_metric(key, grad_norm, step_num)
                logged[key] = grad_norm

        if log_per_layer:
            for layer, norms in layer_norms.items():
                layer_norm = sum(norms) ** (1.0 / norm_type)
                key = f"{prefix}/layer/{layer}"
                self.log_metric(key, layer_norm, step_num)
                logged[key] = layer_norm

        if log_global and global_norms:
            global_norm = sum(global_norms) ** (1.0 / norm_type)
            key = f"{prefix}/global"
            self.log_metric(key, global_norm, step_num)
            logged[key] = global_norm

        return logged

    def log_grad_stats(
        self,
        model: nn.Module,
        *,
        step_num: int | None = None,
        prefix: str = "grad_stats",
    ) -> dict[str, float]:
        """Log per-parameter gradient mean, std, and max in addition to the L2 norm.

        Useful for diagnosing dead neurons, exploding directions, or
        pathological gradient distributions that a single norm misses.

        Returns
        -------
        dict[str, float]
            All logged values keyed by metric name.
        """
        logged: dict[str, float] = {}

        for param_name, param in model.named_parameters():
            if param.grad is None:
                continue

            g = param.grad.detach()
            stats = {
                "norm":  g.norm(2).item(),
                "mean":  g.mean().item(),
                "std":   g.std().item(),
                "max":   g.abs().max().item(),
            }
            for stat_name, stat_val in stats.items():
                key = f"{prefix}/{param_name}/{stat_name}"
                self.log_metric(key, stat_val, step_num)
                logged[key] = stat_val

        return logged

    def log_weights(
        self,
        model: nn.Module,
        *,
        step_num: int | None = None,
        prefix: str = "weight",
        norm_type: float = 2.0,
    ) -> dict[str, float]:
        """Log the L\ *p* norm of every parameter tensor.

        Useful for tracking weight growth / collapse during training.
        """
        logged: dict[str, float] = {}
        for param_name, param in model.named_parameters():
            norm = param.data.detach().norm(norm_type).item()
            key = f"{prefix}/{param_name}"
            self.log_metric(key, norm, step_num)
            logged[key] = norm
        return logged

    def log_activations(
        self,
        activations: dict[str, torch.Tensor],
        *,
        step_num: int | None = None,
        prefix: str = "activation",
    ) -> dict[str, float]:
        """Log summary statistics for a dict of named activation tensors.

        Typical usage with forward hooks::

            hooks = {}
            def make_hook(name):
                def hook(module, input, output):
                    acts[name] = output.detach()
                return hook

            for name, layer in model.named_modules():
                hooks[name] = layer.register_forward_hook(make_hook(name))

            # ... forward pass ...
            logger.log_activations(acts, step_num=step)
        """
        logged: dict[str, float] = {}
        for act_name, tensor in activations.items():
            t = tensor.detach().float()
            stats = {
                "mean":      t.mean().item(),
                "std":       t.std().item(),
                "abs_max":   t.abs().max().item(),
                "frac_dead": (t.abs() < 1e-6).float().mean().item(),
            }
            for stat_name, val in stats.items():
                key = f"{prefix}/{act_name}/{stat_name}"
                self.log_metric(key, val, step_num)
                logged[key] = val
        return logged

    def log_policy(
        self,
        *,
        entropy: float | torch.Tensor | None = None,
        kl_div: float | torch.Tensor | None = None,
        clip_frac: float | torch.Tensor | None = None,
        approx_kl: float | torch.Tensor | None = None,
        step_num: int | None = None,
        prefix: str = "policy",
    ) -> None:
        """Log standard PPO / policy-gradient diagnostics."""
        fields = {
            "entropy":   entropy,
            "kl_div":    kl_div,
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
        }
        for name, val in fields.items():
            if val is not None:
                self.log_metric(f"{prefix}/{name}", val, step_num)

    def log_value(
        self,
        *,
        loss: float | torch.Tensor | None = None,
        explained_var: float | torch.Tensor | None = None,
        mean: float | torch.Tensor | None = None,
        std: float | torch.Tensor | None = None,
        step_num: int | None = None,
        prefix: str = "value",
    ) -> None:
        """Log value-function diagnostics.

        ``explained_var`` follows the standard formula
        ``1 - Var(returns - values) / Var(returns)`` and is the single most
        informative signal for value-function quality.
        """
        fields = {
            "loss":         loss,
            "explained_var": explained_var,
            "mean":         mean,
            "std":          std,
        }
        for name, val in fields.items():
            if val is not None:
                self.log_metric(f"{prefix}/{name}", val, step_num)

    def log_rewards(
        self,
        rewards: torch.Tensor | Iterable[float],
        *,
        step_num: int | None = None,
        prefix: str = "reward",
    ) -> None:
        """Log summary statistics over a batch of scalar rewards."""
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(list(rewards), dtype=torch.float32)
        r = rewards.detach().float()
        self.log_metric(f"{prefix}/mean",   r.mean().item(),  step_num)
        self.log_metric(f"{prefix}/std",    r.std().item(),   step_num)
        self.log_metric(f"{prefix}/min",    r.min().item(),   step_num)
        self.log_metric(f"{prefix}/max",    r.max().item(),   step_num)

    def log_disc(
        self,
        *,
        real_scores: torch.Tensor | None = None,
        fake_scores: torch.Tensor | None = None,
        loss: float | torch.Tensor | None = None,
        gradient_penalty: float | torch.Tensor | None = None,
        step_num: int | None = None,
        prefix: str = "disc",
    ) -> None:
        """Log AMP / adversarial discriminator diagnostics.

        Score tensors are summarised as mean ± std so you can track mode
        collapse, saturation, or reward hacking in one glance.
        """
        if real_scores is not None:
            r = real_scores.detach().float()
            self.log_metric(f"{prefix}/real_mean", r.mean().item(), step_num)
            self.log_metric(f"{prefix}/real_std",  r.std().item(),  step_num)
        if fake_scores is not None:
            f = fake_scores.detach().float()
            self.log_metric(f"{prefix}/fake_mean", f.mean().item(), step_num)
            self.log_metric(f"{prefix}/fake_std",  f.std().item(),  step_num)
        if loss is not None:
            self.log_metric(f"{prefix}/loss", loss, step_num)
        if gradient_penalty is not None:
            self.log_metric(f"{prefix}/gradient_penalty", gradient_penalty, step_num)

    def log_advantages(
        self,
        advantages: torch.Tensor,
        *,
        step_num: int | None = None,
        prefix: str = "advantage",
    ) -> None:
        """Log advantage statistics — mean ≈ 0 and std ≈ 1 after normalisation."""
        a = advantages.detach().float()
        self.log_metric(f"{prefix}/mean", a.mean().item(), step_num)
        self.log_metric(f"{prefix}/std",  a.std().item(),  step_num)
        self.log_metric(f"{prefix}/min",  a.min().item(),  step_num)
        self.log_metric(f"{prefix}/max",  a.max().item(),  step_num)

    def log_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        step_num: int | None = None,
        prefix: str = "optim",
        log_lr: bool = True,
        log_adam_stats: bool = False,
    ) -> None:
        """Log optimizer state — learning rate and optionally Adam moments.

        Parameters
        ----------
        log_adam_stats:
            When ``True`` and the optimizer is an Adam variant, emit the
            per-group RMS of the first and second moment estimates.  These
            reveal whether Adam's effective step size is collapsing or
            diverging.
        """
        for i, pg in enumerate(optimizer.param_groups):
            tag = f"group{i}" if len(optimizer.param_groups) > 1 else "all"

            if log_lr:
                self.log_metric(f"{prefix}/lr/{tag}", pg["lr"], step_num)

            if log_adam_stats and "exp_avg" in (optimizer.state.get(next(iter(pg["params"])), {})):
                m1_norms, m2_norms = [], []
                for p in pg["params"]:
                    state = optimizer.state.get(p)
                    if state is None:
                        continue
                    if "exp_avg" in state:
                        m1_norms.append(state["exp_avg"].norm().item())
                    if "exp_avg_sq" in state:
                        m2_norms.append(state["exp_avg_sq"].norm().item())
                if m1_norms:
                    self.log_metric(
                        f"{prefix}/adam_m1_norm/{tag}",
                        sum(m1_norms) / len(m1_norms),
                        step_num,
                    )
                if m2_norms:
                    self.log_metric(
                        f"{prefix}/adam_m2_norm/{tag}",
                        sum(m2_norms) / len(m2_norms),
                        step_num,
                    )

    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step_num: int | None = None,
    ) -> int:
        """Log a parameter / gradient distribution as a TensorBoard histogram.

        Histograms complement scalar norms by surfacing bimodal distributions,
        heavy tails, or dead-unit spikes that scalars mask.
        """
        self._metric_steps[name] += 1
        effective_step = step_num if step_num is not None else self._metric_steps[name]
        self.writer.add_histogram(name, values.detach(), effective_step)
        return effective_step

    def log_grad_histograms(
        self,
        model: nn.Module,
        *,
        step_num: int | None = None,
        prefix: str = "grad_hist",
    ) -> None:
        """Emit gradient histograms for every parameter that has a gradient."""
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(
                    f"{prefix}/{param_name}",
                    param.grad,
                    step_num,
                )

    def log_weight_histograms(
        self,
        model: nn.Module,
        *,
        step_num: int | None = None,
        prefix: str = "weight_hist",
    ) -> None:
        """Emit weight histograms for every parameter tensor."""
        for param_name, param in model.named_parameters():
            self.log_histogram(
                f"{prefix}/{param_name}",
                param.data,
                step_num,
            )

    @staticmethod
    def _format_wall_time(wall_time: float) -> str:
        if wall_time < 60:
            return f"{wall_time:.1f}s"
        if wall_time < 3600:
            return f"{int(wall_time // 60)}m {wall_time % 60:.0f}s"
        return f"{int(wall_time // 3600)}h {int((wall_time % 3600) // 60)}m"

    def pprint(
        self,
        iteration: int | None = None,
        wall_time: float | None = None,
        samples: int | None = None,
        *,
        title: str | None = None,
        show_steps: bool = False,
    ) -> None:
        """Print a two-column metrics table for the current iteration or eval.

        Parameters
        ----------
        show_steps:
            When ``True``, append each metric's internal step counter next to
            its averaged value so you can verify call-site consistency.
        """
        L, R = self._L, self._R
        W = L + R + 5  # ║ {L} ║ {R} ║

        def fmt_val(v: float) -> str:
            if v != 0 and (abs(v) >= 1000 or abs(v) < 0.001):
                return f"{v:{R}.3e}"
            return f"{v:{R}.4f}"

        def data_row(label: str, value: str) -> str:
            return f"║ {label:<{L}} ║ {value:>{R}} ║"

        def hline() -> str:
            return f"╠{'═' * (L + 2)}╬{'═' * (R + 2)}╣"

        def group_line(group_title: str) -> str:
            inner = f" {group_title} "
            pad_right = max(0, L + 2 - 2 - len(inner))
            return f"╠{'═' * 2}{inner}{'═' * pad_right}╬{'═' * (R + 2)}╣"

        # Average and flush the accumulated metrics
        metrics: dict[str, float] = {
            k: sum(vs) / len(vs) for k, vs in self._metric_accum.items()
        }
        self._metric_accum.clear()

        if title is None:
            if iteration is None:
                raise ValueError("pprint requires either title or iteration")
            title = f"Iteration {iteration:,}"

        lines: list[str] = []
        lines.append(f"╔{'═' * (W - 2)}╗")
        lines.append(f"║{title:^{W - 2}}║")
        lines.append(hline())

        if wall_time is not None:
            lines.append(data_row("Wall Time", self._format_wall_time(wall_time)))
        if samples is not None:
            lines.append(data_row("Samples", f"{samples:,}"))

        # Group metrics by prefix
        grouped: dict[str, list[tuple[str, float]]] = {}
        for k, v in metrics.items():
            prefix = k.split("/")[0] if "/" in k else "__other__"
            grouped.setdefault(prefix, []).append((k, v))

        shown = [g for g in self._GROUP_ORDER if g in grouped]
        shown += [g for g in grouped if g not in self._GROUP_ORDER and g != "__other__"]
        if "__other__" in grouped:
            shown.append("__other__")

        for group in shown:
            lines.append(group_line(group if group != "__other__" else "other"))
            for key, val in grouped[group]:
                label = key.split("/", 1)[1] if "/" in key else key
                if show_steps:
                    step_str = f"[{self._metric_steps.get(key, 0):,}]"
                    # Truncate label to leave room for the step annotation
                    max_label = L - len(step_str) - 1
                    label = f"{label[:max_label]} {step_str}"
                lines.append(data_row(label, fmt_val(val)))

        lines.append(f"╚{'═' * (L + 2)}╩{'═' * (R + 2)}╝")
        table = "\n".join(lines)
        print(table)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
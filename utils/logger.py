import logging
import os

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
        self.step_num = 0
        self.epoch_num = 0

        self._metric_accum: dict[str, list[float]] = {}

        # ---- W&B ----
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

    def step(self):
        self.step_num += 1

    def epoch(self):
        self.epoch_num += 1
        if self.use_wandb:
            self.wandb.log({"epoch": self.epoch_num}, step=self.step_num)

    def info(self, msg: str):
        self._logger.info(msg)

    def warn(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def log_metric(self, name: str, value: float):
        if ArrayUtils.is_tensor(value):
            value = value.detach().item()

        if self.verbose:
            self.info(f"{name}: {value} (step={self.step_num})")

        self._metric_accum.setdefault(name, []).append(value)

        # TensorBoard
        self.writer.add_scalar(name, value, self.step_num)

        # W&B
        if self.use_wandb:
            self.wandb.log({name: value}, step=self.step_num)

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
    ) -> None:
        """Print a two-column metrics table for the current iteration or eval."""
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

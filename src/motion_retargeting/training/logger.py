# training/logger.py

from __future__ import annotations

import math
import shutil
from pathlib import Path

SECTION_INDENT = "    "
TERM_INDENT = "      "


def format_value(value: float) -> str:
    """ 4 decimals; scientific when that would hide the magnitude; nan / inf / -inf as such. """
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value == 0 or 1e-3 <= abs(value) < 1e6:
        return f"{value:.4f}"
    return f"{value:.3e}"


class MetricLogger:
    """
    Minimal metric logger:
      - accumulation / averaging: `update` adds a step's metrics to running sums,
        `average` returns the per-key means since the last `average` and resets them
      - storage: every `write` is kept in `history` as {key: [(step, value), ...]}
      - TensorBoard (on by default): every `write` is also sent to a SummaryWriter
      - display: `display` prints a formatted metrics table, only when `verbose`
    """

    def __init__(
        self,
        log_dir: str | Path = "runs",
        use_tensorboard: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.history: dict[str, list[tuple[int, float]]] = {}

        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

        self.writer = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def update(self, metrics: dict[str, float]) -> None:
        """ Add one step's metrics to the running sums. """
        for key, value in metrics.items():
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._counts[key] = self._counts.get(key, 0) + 1

    def average(self) -> dict[str, float]:
        """ Per-key mean of everything `update`d since the last call; then reset. """
        means = {key: total / self._counts[key] for key, total in self._sums.items()}
        self._sums.clear()
        self._counts.clear()
        return means

    def write(self, metrics: dict[str, float], step: int) -> None:
        """ Store the metrics at `step` in `history` and, if enabled, TensorBoard. """
        for key, value in metrics.items():
            value = float(value)
            self.history.setdefault(key, []).append((step, value))
            if self.writer is not None:
                self.writer.add_scalar(key, value, step)

    def display(self, header: str, metrics: dict[str, float]) -> None:
        """ Print `header` followed by the metrics table, if `verbose`. """
        if self.verbose:
            print(f"{header}\n{self.format_metrics(metrics)}")

    @staticmethod
    def format_metrics(metrics: dict[str, float], width: int | None = None) -> str:
        """
        Metrics grouped into sections by the part of the key before the last "/", one
        "term .... value" row per metric, values right-aligned in one column, e.g.
        {"group_a/reconstruction": 0.12, "grad/group_a/decoder": 3e-06} ->

            group_a
              reconstruction ..    0.1200
            grad/group_a
              decoder ......... 3.000e-06

        Sections keep the order their keys first appear in; keys without a "/" go in a
        "metrics" section. Lines fit in `width` columns (default: the terminal width, or
        80 when output is redirected): a term too long for its row is printed whole on
        its own line, with its value on the dotted line below. Names are never truncated.
        """
        if not metrics:
            return "    (no metrics)"
        if width is None:
            width = shutil.get_terminal_size().columns

        sections: dict[str, list[tuple[str, str]]] = {}
        for key, value in metrics.items():
            section, _, term = key.rpartition("/")
            sections.setdefault(section or "metrics", []).append((term, format_value(value)))

        rows = [row for section_rows in sections.values() for row in section_rows]
        value_width = max(len(value) for _, value in rows)
        # Term plus its " .." leader, as wide as the longest term needs or the width allows.
        field = min(
            max(len(term) for term, _ in rows) + 3,
            width - len(TERM_INDENT) - 1 - value_width,
        )

        lines = []
        for section, section_rows in sections.items():
            lines.append(SECTION_INDENT + section)
            for term, value in section_rows:
                value = f"{value:>{value_width}}"
                if len(term) + 3 <= field:
                    lines.append(f"{TERM_INDENT}{term} {'.' * (field - len(term) - 1)} {value}")
                else:
                    lines.append(TERM_INDENT + term)
                    lines.append(f"{TERM_INDENT}  {'.' * max(field - 2, 1)} {value}")
        return "\n".join(lines)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

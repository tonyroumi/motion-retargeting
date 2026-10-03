# training/logger.py

from __future__ import annotations

from pathlib import Path


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
            print(f"{header} {self.format_metrics(metrics)}")

    @staticmethod
    def format_metrics(metrics: dict[str, float]) -> str:
        """
        Table of metrics with one column per group (the part of the key before "/") and
        one row per term, e.g. {"group_a/reconstruction": 0.12, "total/generator": 0.5} ->

                               group_a     total
            reconstruction      0.1200
            generator                     0.5000

        Keys without a "/" go in a "metrics" column; missing (group, term) cells stay blank.
        """
        if not metrics:
            return "(no metrics)"

        table: dict[str, dict[str, float]] = {}  # term -> group -> value
        groups: list[str] = []
        for key, value in metrics.items():
            group, _, term = key.rpartition("/")
            group = group or "metrics"
            if group not in groups:
                groups.append(group)
            table.setdefault(term, {})[group] = value

        term_width = max(len(term) for term in table)
        col_width = max(10, *(len(group) for group in groups))

        header = " " * term_width + "".join(f"  {group:>{col_width}}" for group in groups)
        rows = [
            f"{term:<{term_width}}" + "".join(
                f"  {values[group]:>{col_width}.4f}" if group in values else "  " + " " * col_width
                for group in groups
            )
            for term, values in table.items()
        ]

        return "\n" + "\n".join("    " + line.rstrip() for line in [header, *rows])

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

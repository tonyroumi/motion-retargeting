# training/trainer.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from motion_retargeting.methods.base import RetargetingMethod
from motion_retargeting.training.utils import move_to_device

from .logger import MetricLogger


@dataclass
class TrainerConfig:
    num_epochs: int = 100

    log_interval: int = 10
    validation_interval: int = 10
    checkpoint_interval: int = 10

    # Each run writes to <log_dir>/<timestamp>/: TensorBoard files at the top level,
    # checkpoints in checkpoints/, validation visualizations in checkpoints/visualizations/.
    log_dir: str = "logs"
    use_tensorboard: bool = True
    verbose: bool = True

    device: str = "cuda"
    max_steps: int | None = None  # Optional exact optimizer-update budget, independent of corpus size.
    checkpoint_interval_steps: int | None = None


class Trainer:
    """ Generic training loop. """

    def __init__(
        self,
        method: RetargetingMethod,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: TrainerConfig,
    ):
        self.method = method

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.current_epoch = 0
        self.global_step = 0
        self.epoch_complete = True
        for name in ("max_steps", "checkpoint_interval_steps"):
            value = getattr(cfg, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")

        self.run_dir = Path(cfg.log_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.visualization_dir = self.checkpoint_dir / "visualizations"
        self.checkpoint_dir.mkdir(parents=True)

        self.logger = MetricLogger(
            log_dir=self.run_dir,
            use_tensorboard=cfg.use_tensorboard,
            verbose=cfg.verbose,
        )

    def train(self) -> None:
        for epoch in range(self.current_epoch, self.cfg.num_epochs):
            if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                break
            self.current_epoch = epoch

            train_metrics = self.train_epoch()

            self.logger.display(f"[Epoch {epoch + 1}/{self.cfg.num_epochs}]", train_metrics)

            if (
                self.val_loader is not None
                and (epoch + 1) % self.cfg.validation_interval == 0
            ):
                val_metrics = self.validate_epoch()
                self.logger.write(val_metrics, self.global_step)

                self.logger.display("[Validation]", val_metrics)

                self.visualize(self.visualization_dir / f"epoch_{epoch + 1}")

            if (epoch + 1) % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch + 1}.pt", epoch_complete=self.epoch_complete,
                )

        self.save_checkpoint(
            self.checkpoint_dir / "final.pt", epoch_complete=self.epoch_complete,
        )

        self.logger.close()

    def train_epoch(self) -> dict[str, float]:

        self.method.train()

        self.epoch_complete = False
        for index, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.device)

            metrics = self.method.train_step(batch)

            self.global_step += 1
            self.logger.update(metrics)
            self.logger.write(metrics, self.global_step)

            if self.global_step % self.cfg.log_interval == 0:
                self.logger.display(f"  step={self.global_step}", metrics)
            self.epoch_complete = index + 1 == len(self.train_loader)
            if self.cfg.checkpoint_interval_steps and self.global_step % self.cfg.checkpoint_interval_steps == 0:
                self.save_checkpoint(self.checkpoint_dir / f"step_{self.global_step}.pt", epoch_complete=self.epoch_complete)
            if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                break

        return self.logger.average()

    @torch.no_grad()
    def validate_epoch(self) -> dict[str, float]:

        self.method.eval()

        for batch in self.val_loader:
            batch = move_to_device(batch, self.device)

            metrics = self.method.validation_step(batch)

            self.logger.update(metrics)

        return self.logger.average()

    @torch.no_grad()
    def visualize(self, save_dir: Path) -> None:
        """ Let the method save visualizations of the first validation batch into `save_dir`. """
        self.method.eval()
        batch = move_to_device(next(iter(self.val_loader)), self.device)
        self.method.visualize(batch, save_dir)

    def save_checkpoint(
        self,
        path: str | Path,
        epoch_complete: bool = True,
    ) -> None:
        path = Path(path)
        
        checkpoint = {
            "epoch_complete": epoch_complete,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "method": self.method.state_dict(),
        }

        torch.save(checkpoint, path)

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.method.load_state_dict(checkpoint["method"])

        self.epoch_complete = checkpoint.get("epoch_complete", True)
        self.current_epoch = checkpoint["epoch"] + int(self.epoch_complete)
        self.global_step = checkpoint["global_step"]

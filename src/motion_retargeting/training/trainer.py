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
                    self.checkpoint_dir / f"epoch_{epoch + 1}.pt"
                )

        self.save_checkpoint(
            self.checkpoint_dir / "final.pt"
        )

        self.logger.close()

    def train_epoch(self) -> dict[str, float]:

        self.method.train()

        for batch in self.train_loader:
            batch = move_to_device(batch, self.device)

            metrics = self.method.train_step(batch)

            self.global_step += 1
            self.logger.update(metrics)
            self.logger.write(metrics, self.global_step)

            if self.global_step % self.cfg.log_interval == 0:
                self.logger.display(f"  step={self.global_step}", metrics)

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
    ) -> None:
        path = Path(path)
        
        checkpoint = {
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

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]

from __future__ import annotations

import argparse

import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from motion_retargeting.data.dataset import compute_stats, split_dataset
from motion_retargeting.data.factory import build_datasets
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.training.trainer import Trainer, TrainerConfig
from motion_retargeting.training.utils import ZippedLoaders


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True,
)

args = parser.parse_args()


def main():
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])

    datasets = build_datasets(cfg["dataset"])

    trainer_cfg = cfg["trainer"]

    # Split every domain, then normalize all of its splits with stats from its train split only.
    train_datasets, val_datasets, test_datasets = {}, {}, {}
    for name, dataset in datasets.items():
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            val_fraction=trainer_cfg["val_fraction"],
            test_fraction=trainer_cfg["test_fraction"],
        )
        vel_mean, vel_std, ang_mean, ang_std = compute_stats(train_dataset)

        train_dataset.set_normalization(vel_mean, vel_std, ang_mean, ang_std)
        val_dataset.set_normalization(vel_mean, vel_std, ang_mean, ang_std)
        test_dataset.set_normalization(vel_mean, vel_std, ang_mean, ang_std)

        train_datasets[name] = train_dataset
        val_datasets[name] = val_dataset
        test_datasets[name] = test_dataset

    # Draw the same number of samples from every domain each epoch (a fresh random subset
    # of the larger ones), so the zipped loaders always yield equally sized batches.
    num_samples = min(len(dataset) for dataset in train_datasets.values())
    train_loaders = {
        name: DataLoader(
            dataset,
            batch_size=trainer_cfg["batch_size"],
            sampler=RandomSampler(dataset, num_samples=num_samples),
            pin_memory=True,
        )
        for name, dataset in train_datasets.items()
    }

    # Validation: the same fixed, equally sized, unshuffled windows from every domain.
    num_val_samples = min(len(dataset) for dataset in val_datasets.values())
    val_loaders = {
        name: DataLoader(
            Subset(dataset, range(num_val_samples)),
            batch_size=trainer_cfg["batch_size"],
            shuffle=False,
            pin_memory=True,
        )
        for name, dataset in val_datasets.items()
    }

    specs = {
        name: DomainSpec.from_dataset(dataset)
        for name, dataset in train_datasets.items()
    }

    method = PANMethod(
        model_cfg=PANModelConfig.from_dict(cfg["method"]["model"]),
        domain_specs=specs,
        learning_rate=trainer_cfg["learning_rate"],
        device=device,
        cfg=PANConfig(**cfg["method"].get("params", {})),
    )

    trainer = Trainer(
        method=method,
        train_loader=ZippedLoaders(train_loaders),
        val_loader=ZippedLoaders(val_loaders),
        cfg=TrainerConfig(
            num_epochs=trainer_cfg["num_epochs"],
            log_interval=trainer_cfg["log_interval"],
            validation_interval=trainer_cfg["validation_interval"],
            checkpoint_interval=trainer_cfg["checkpoint_interval"],
            log_dir=trainer_cfg["log_dir"],
            use_tensorboard=trainer_cfg["use_tensorboard"],
            verbose=trainer_cfg["verbose"],
            device=str(device),
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()

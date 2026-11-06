from __future__ import annotations

import argparse

import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import build_splits as build_pan_splits
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.san.config import DomainSpec as SANDomainSpec, SANConfig, SANModelConfig
from motion_retargeting.methods.san.data import build_san_splits
from motion_retargeting.methods.san.method import SANMethod
from motion_retargeting.training.trainer import Trainer, TrainerConfig
from motion_retargeting.training.utils import ZippedLoaders


# method name -> (domain spec, model config, method config, method) classes, and its dataset splits
METHODS = {
    "pan": (DomainSpec, PANModelConfig, PANConfig, PANMethod, build_pan_splits),
    "san": (SANDomainSpec, SANModelConfig, SANConfig, SANMethod, build_san_splits),
}


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

    trainer_cfg = cfg["trainer"]

    spec_cls, model_cfg_cls, method_cfg_cls, method_cls, build_splits = METHODS[cfg["method"]["name"]]

    # Built from the raw collections: each method reads edge rotations and the absolute root.
    train_datasets, val_datasets = build_splits(cfg["dataset"])

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
        name: spec_cls.from_dataset(dataset)
        for name, dataset in train_datasets.items()
    }

    method = method_cls(
        model_cfg=model_cfg_cls.from_dict(cfg["method"]["model"]),
        domain_specs=specs,
        learning_rate=trainer_cfg["learning_rate"],
        device=device,
        cfg=method_cfg_cls(**cfg["method"].get("params", {})),
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

from __future__ import annotations

import argparse

import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from motion_retargeting.data.dataset import compute_stats, split_dataset
from motion_retargeting.data.factory import build_datasets
from motion_retargeting.methods.pan.config import (
    DomainSpec,
    PANConfig,
    PANModelConfig,
    PANReferenceConfig,
    PANReferenceDomainSpec,
    PANReferenceModelConfig,
    REFERENCE_RECIPE,
)
from motion_retargeting.methods.pan.data import build_reference_splits
from motion_retargeting.methods.pan.method import PANMethod, PANReferenceMethod
from motion_retargeting.methods.san.config import DomainSpec as SANDomainSpec, SANConfig, SANModelConfig
from motion_retargeting.methods.san.method import SANMethod
from motion_retargeting.training.trainer import Trainer, TrainerConfig
from motion_retargeting.training.utils import ZippedLoaders


# method name -> (domain spec, model config, method config, method) classes
METHODS = {
    "pan": (DomainSpec, PANModelConfig, PANConfig, PANMethod),
    "san": (SANDomainSpec, SANModelConfig, SANConfig, SANMethod),
}

# Selected by `method.params.recipe: mixamo_reference` under method name "pan".
PAN_REFERENCE_METHOD = (PANReferenceDomainSpec, PANReferenceModelConfig, PANReferenceConfig, PANReferenceMethod)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True,
)

args = parser.parse_args()


def uses_reference_recipe(method_cfg: dict) -> bool:
    """ Whether the config selects PAN's reference-Mixamo recipe; PAN stays legacy by default. """
    return method_cfg["name"] == "pan" and (method_cfg.get("params") or {}).get("recipe") == REFERENCE_RECIPE


def main():
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])

    trainer_cfg = cfg["trainer"]

    if uses_reference_recipe(cfg["method"]):
        # Built from the raw collections: the shared windows below have no absolute root or edges.
        train_datasets, val_datasets = build_reference_splits(cfg["dataset"])
    else:
        datasets = build_datasets(cfg["dataset"])

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

    spec_cls, model_cfg_cls, method_cfg_cls, method_cls = (
        PAN_REFERENCE_METHOD if uses_reference_recipe(cfg["method"]) else METHODS[cfg["method"]["name"]]
    )

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

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from motion_retargeting.methods.pan.config import DomainSpec as PANDomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import build_pan_splits
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.san.config import DomainSpec as SANDomainSpec, SANConfig, SANModelConfig
from motion_retargeting.methods.san.data import build_san_splits
from motion_retargeting.methods.san.method import SANMethod

from motion_retargeting.training.trainer import Trainer, TrainerConfig
from motion_retargeting.training.utils import PrimaryRecycledLoaders, ZippedLoaders


METHODS = {
    "pan": (PANDomainSpec, PANModelConfig, PANConfig, PANMethod, build_pan_splits),
    "san": (SANDomainSpec, SANModelConfig, SANConfig, SANMethod, build_san_splits),
}


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True,
)
parser.add_argument("--device", help="Override the configured device (e.g. cpu for preparation)")
parser.add_argument("--prepare-only", action="store_true", help="Build/validate datasets and model, snapshot the run, then exit without training")

def main(argv=None):
    args = parser.parse_args(argv)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.device:
        cfg["device"] = args.device

    if cfg.get("seed") is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device(cfg["device"])

    trainer_cfg = cfg["trainer"]

    spec_cls, model_cfg_cls, method_cfg_cls, method_cls, build_splits = METHODS[cfg["method"]["name"]]

    train_datasets, val_datasets = build_splits(cfg["dataset"])

    sampling = trainer_cfg.get("sampling", "balanced_min")
    if sampling == "primary_recycle":
        train_loaders = {
            name: DataLoader(dataset, batch_size=trainer_cfg["batch_size"], shuffle=True,
                             drop_last=True, pin_memory=device.type == "cuda",
                             generator=torch.Generator().manual_seed(cfg["seed"] + i) if cfg.get("seed") is not None else None)
            for i, (name, dataset) in enumerate(train_datasets.items())
        }
        train_loader = PrimaryRecycledLoaders(train_loaders, trainer_cfg["primary_domain"])
    elif sampling == "balanced_min":
        # Legacy sampling remains unchanged for existing configurations.
        num_samples = min(len(dataset) for dataset in train_datasets.values())
        train_loaders = {
            name: DataLoader(dataset, batch_size=trainer_cfg["batch_size"],
                             sampler=RandomSampler(dataset, num_samples=num_samples), pin_memory=True)
            for name, dataset in train_datasets.items()
        }
        train_loader = ZippedLoaders(train_loaders)
    else:
        raise ValueError(f"Unknown training sampling mode {sampling!r}")

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
        train_loader=train_loader,
        val_loader=ZippedLoaders(val_loaders),
        cfg=TrainerConfig(
            num_epochs=trainer_cfg["num_epochs"],
            max_steps=trainer_cfg.get("max_steps"),
            checkpoint_interval_steps=trainer_cfg.get("checkpoint_interval_steps"),
            log_interval=trainer_cfg["log_interval"],
            validation_interval=trainer_cfg["validation_interval"],
            checkpoint_interval=trainer_cfg["checkpoint_interval"],
            log_dir=trainer_cfg["log_dir"],
            use_tensorboard=trainer_cfg["use_tensorboard"],
            verbose=trainer_cfg["verbose"],
            device=str(device),
        ),
    )

    # Snapshot the actual experiment instead of relying on subsequently edited YAML.
    (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "status", "--porcelain"], text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    provenance = {"revision": revision, "revision_dirty": dirty, "torch_version": torch.__version__,
                  "numpy_version": np.__version__, "sampling": sampling,
                  "steps_per_epoch": len(train_loader), "max_steps": trainer_cfg.get("max_steps"), "domains": {
                      name: {"train_windows": len(data), "test_windows": len(val_datasets[name]),
                             "data": getattr(data, "provenance", None)} for name, data in train_datasets.items()}}
    provenance["prepare_only"] = args.prepare_only
    (trainer.run_dir / "data_provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.prepare_only:
        print(f"Prepared {trainer.run_dir}: {len(train_loader)} full batches per epoch; no optimizer steps executed")
        trainer.logger.close()
        return
    trainer.train()


if __name__ == "__main__":
    main()

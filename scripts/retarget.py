"""
Retarget motion between the two domains of a trained PAN checkpoint, in either or both directions.

For each direction the source motion is saved first and the retargeted motion second, in a folder of
their own that must not already hold either file:

    <output-dir>/<source>_to_<target>/00_original_<source>.npz
    <output-dir>/<source>_to_<target>/01_retargeted_<target>.npz

Each .npz (plain arrays, loadable with `numpy.load` without pickle) holds:
    positions        [T, J_fk, 3]  world joint positions from the model's FK; the root starts at the
                                   origin and is integrated from the root velocity, so absolute root
                                   height and start position are not represented
    position_parents [J_fk]        parent of each `positions` joint (reference archives add End Sites)
    rotations        [T, J, 4]     local joint quaternions (w, x, y, z), normalized; joint 0 is the root
    rotation_parents [J]
    root_velocity    [T, 3]        root displacement per frame
    yaw_velocity     [T]           root yaw increment per frame
    channels         [T, 4J + 4]   the denormalized network channels the above are read from
    offsets          [J_fk, 3]     the skeleton bank's offsets used by FK
    fps, domain, bank_id, role ("original" / "retargeted"), source_domain, source_bank_id,
    checkpoint_physical_units      the units the checkpoint metadata declares for offsets and positions

The source is one window of a dataset split (default: the first holdout window), or with --clip one
whole adapter clip (raw sources only), run in a single pass without window stitching: the generator's
attention is per frame and its temporal layers are convolutions. Their receptive field is wider than a
training window, so a clip's frames see real neighbors where a window's see reflect padding, and the
two differ. A clip is resampled to the group's `target_fps` and trimmed to a multiple of 4 frames.
The target skeleton is one of the target domain's skeleton banks, chosen with --target-bank.

--config must window the data exactly as the checkpoint's training run did (same sources, sequence
length, stride, frame rate): the script refuses a checkpoint whose banks or statistics differ.

    python scripts/retarget.py --config configs/methods/pan.yaml --checkpoint <run>/checkpoints/final.pt \
        --output-dir outputs/retarget
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from motion_retargeting.data.adapters.factory import build_adapter
from motion_retargeting.kinematics.quaternions import quat_normalize
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import (
    PANDataConfig,
    PANMotionDataset,
    build_pan_splits,
    resample_indices,
    to_joint_channels,
)
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.model import PANDomain, PANModel


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget motion with a trained PAN checkpoint.")
    parser.add_argument("--config", required=True, help="The training YAML the checkpoint was trained with")
    parser.add_argument("--checkpoint", required=True, help="A checkpoint saved by the trainer (.pt)")
    parser.add_argument("--output-dir", required=True, help="Created if missing; existing outputs are never overwritten")
    parser.add_argument("--source", default="both", help="Source group (e.g. group_a), or 'both' for both directions")
    parser.add_argument("--split", choices=("holdout", "train"), default="holdout", help="Split to take the window from")
    parser.add_argument("--index", type=int, default=0, help="Window index in the split (with --clip: clip index)")
    parser.add_argument("--clip", action="store_true", help="Retarget a whole adapter clip instead of one window")
    parser.add_argument("--target-bank", type=int, default=0, help="Skeleton bank of the target domain to retarget onto")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def load_method(cfg: Dict[str, Any], checkpoint: str, specs: Dict[str, DomainSpec], device: torch.device) -> PANMethod:
    method = PANMethod(
        model_cfg=PANModelConfig.from_dict(cfg["method"]["model"]),
        domain_specs=specs,
        device=device,
        cfg=PANConfig(**cfg["method"].get("params", {})),
    )
    method.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["method"])
    method.eval()
    return method


def check_data_matches_checkpoint(banks: Dict[str, Tuple[torch.Tensor, ...]], model: PANModel) -> None:
    """
    banks: {domain: (offsets, mean, std)} of the data, copied before loading, since the model's buffers
    can share the data's tensors and loading writes into them. Windows are normalized with the data's
    statistics, so they must be the checkpoint's.
    """
    for name, (offsets, mean, std) in banks.items():
        domain = model.domains[name]
        if not (
            offsets.shape == domain.offsets.shape
            and torch.equal(offsets.to(domain.offsets), domain.offsets)
            and torch.allclose(mean.to(domain.mean), domain.mean)
            and torch.allclose(std.to(domain.std), domain.std)
        ):
            raise ValueError(f"{name}: the data's skeleton banks or statistics differ from the checkpoint's")


def window_motion(dataset: PANMotionDataset, index: int) -> Tuple[torch.Tensor, int]:
    """ Normalized window [W, 4J + 4] and its bank. """
    if not 0 <= index < len(dataset):
        raise IndexError(f"Window index {index} is outside the split's {len(dataset)} windows")
    item = dataset[index]
    return item["motion"], int(item["bank_id"])


def clip_motion(group_cfg: Dict[str, Any], index: int, domain: PANDomain) -> Tuple[torch.Tensor, int, float]:
    """ A whole adapter clip as normalized channels [T, 4J + 4], its bank and frame rate. """
    if "reference_archive" in group_cfg:
        raise ValueError("--clip needs adapter sources; reference archives only hold windows")
    config = PANDataConfig(**(group_cfg.get("params") or {}))
    clips = [clip for source in group_cfg["sources"] for clip in build_adapter(source).load().motions]
    if not 0 <= index < len(clips):
        raise IndexError(f"Clip index {index} is outside the group's {len(clips)} clips")
    clip = clips[index]

    rotations, root_position, fps = clip.joint_rotations, clip.root_position, clip.fps
    if config.target_fps is not None and fps != config.target_fps:
        indices = resample_indices(rotations.shape[0], fps, config.target_fps)
        rotations, root_position, fps = rotations[indices], root_position[indices], config.target_fps
    channels = to_joint_channels(rotations, root_position)
    channels = channels[:channels.shape[0] // 4 * 4]  # Two stride-2 encoder layers, two x2 decoder layers
    if channels.shape[0] < 16:
        raise ValueError(f"Clip {index} has too few frames ({channels.shape[0]}) to retarget")

    banks = [k for k in range(domain.offsets.shape[0]) if torch.equal(domain.offsets[k].cpu(), clip.offsets)]
    if not banks:
        raise ValueError(f"Clip {index}'s skeleton is none of the checkpoint's banks (too short for training windows?)")
    bank = banks[0]
    channels = channels.to(domain.mean)
    return (channels - domain.mean[bank]) / domain.std[bank], bank, fps


@torch.no_grad()
def retarget(model: PANModel, source: str, target: str, motion: torch.Tensor, target_bank: int) -> torch.Tensor:
    """ Normalized [1, T, C_source] -> normalized [1, T, C_target], as the generator's cross routes compute it. """
    destination = model.domains[target]
    skeleton_code = destination.body_part_encoder(destination.offsets[target_bank:target_bank + 1])
    return model.decode(target, model.domains[source].motion_encoder(motion), skeleton_code)


@torch.no_grad()
def export(method: PANMethod, name: str, motion: torch.Tensor, bank: int) -> Dict[str, np.ndarray]:
    """ Arrays of one normalized motion [1, T, 4J + 4] on bank `bank` of domain `name`. """
    domain = method.model.domains[name]
    bank_ids = torch.tensor([bank], device=motion.device)
    channels = domain.denormalize(motion, bank_ids)[0]
    num_frames = channels.shape[0]
    rotations = quat_normalize(channels[:, :-4].reshape(num_frames, -1, 4))
    offsets = domain.fk_offsets if domain.is_reference else domain.offsets
    arrays = {
        "positions": domain.fk.world_positions(domain.bones(motion, bank_ids))[0],
        "position_parents": torch.tensor(domain.fk.parents),
        "rotations": rotations,
        "rotation_parents": method.domain_specs[name].parents,
        "root_velocity": channels[:, -4:-1],
        "yaw_velocity": channels[:, -1],
        "channels": channels,
        "offsets": offsets[bank],
    }
    return {key: value.cpu().numpy() for key, value in arrays.items()}


def save(path: Path, arrays: Dict[str, Any]) -> None:
    with open(path, "xb") as file:  # Never overwrite.
        np.savez(file, **arrays)
    print(f"Saved {path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg["method"]["name"] != "pan":
        raise ValueError("This script retargets with PAN checkpoints only")

    groups = list(cfg["dataset"])
    if len(groups) != 2:
        raise ValueError(f"PAN retargeting needs exactly two groups, the config has {groups}")
    if args.source != "both" and args.source not in groups:
        raise ValueError(f"--source must be one of {groups} or 'both', got {args.source!r}")
    sources = groups if args.source == "both" else [args.source]
    directions = [(source, next(group for group in groups if group != source)) for source in sources]

    output_dir = Path(args.output_dir)
    paths = {
        (source, target): (
            output_dir / f"{source}_to_{target}" / f"00_original_{source}.npz",
            output_dir / f"{source}_to_{target}" / f"01_retargeted_{target}.npz",
        )
        for source, target in directions
    }
    existing = [str(path) for pair in paths.values() for path in pair if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    device = torch.device(args.device)
    train_datasets, holdout_datasets = build_pan_splits(cfg["dataset"])
    specs = {name: DomainSpec.from_dataset(dataset) for name, dataset in train_datasets.items()}
    banks = {name: (spec.offsets.clone(), spec.mean.clone(), spec.std.clone()) for name, spec in specs.items()}
    method = load_method(cfg, args.checkpoint, specs, device)
    model = method.model
    check_data_matches_checkpoint(banks, model)
    units = method.checkpoint_metadata()["physical_units"]
    datasets = holdout_datasets if args.split == "holdout" else train_datasets

    # Every direction is computed before anything is written, so a bad index writes nothing.
    outputs = []
    for source, target in directions:
        num_target_banks = model.domains[target].offsets.shape[0]
        if not 0 <= args.target_bank < num_target_banks:
            raise IndexError(f"--target-bank {args.target_bank} is outside {target}'s {num_target_banks} banks")

        if args.clip:
            motion, bank, fps = clip_motion(cfg["dataset"][source], args.index, model.domains[source])
        else:
            motion, bank = window_motion(datasets[source], args.index)
            params = cfg["dataset"][source].get("params") or {}
            fps = 30.0 if "reference_archive" in cfg["dataset"][source] else PANDataConfig(**params).target_fps
        motion = motion[None].to(device)
        fake = retarget(model, source, target, motion, args.target_bank)

        common = {"fps": np.nan if fps is None else fps, "checkpoint_physical_units": units,
                  "source_domain": source, "source_bank_id": bank}
        original_path, retargeted_path = paths[(source, target)]
        outputs.append((original_path, {**export(method, source, motion, bank), **common,
                                        "role": "original", "domain": source, "bank_id": bank}))
        outputs.append((retargeted_path, {**export(method, target, fake, args.target_bank), **common,
                                          "role": "retargeted", "domain": target, "bank_id": args.target_bank}))

    for path, arrays in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        save(path, arrays)


if __name__ == "__main__":
    main()

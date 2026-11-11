"""Bounded, read-only adapter and PAN-coordinate previews (no caches or labeling UI)."""
from pathlib import Path

import numpy as np
import torch

from motion_retargeting.data.adapters.amass import axis_angle_to_quaternion
from motion_retargeting.data.adapters.cmu import amc_to_tensors, parse_amc

from motion_retargeting.data.adapters.factory import build_adapter
from motion_retargeting.data.adapters.bandai import flatten_hierarchy
from motion_retargeting.data.adapters.lafan1_bvh import REFERENCE_HIERARCHIES, read_lafan_reference_bvh
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.kinematics.forward import ForwardKinematics
from motion_retargeting.methods.pan.data import resample_indices, to_joint_channels
from motion_retargeting.methods.pan.human_reference import (
    HUMAN_FK_PARENTS, HUMAN_ROTATION_IDS, human_adapter, human_fk_offsets, human_forward_axis, human_rotation_channels,
)
from motion_retargeting.methods.pan.kinematics import JointForwardKinematics
from motion_retargeting.methods.pan.reference import preprocess_reference_window


@torch.no_grad()
def dataset_panels(source: str, path: Path, space: str = "world", start: int = 0, frames: int = 64,
                   body_models: str | Path | None = None) -> list[dict]:
    """Preview sampled-30-Hz frame ranges, never fitting stats or loading a corpus.

    world: adapter FK, absolute trajectory in metres. training: denormalized PAN
    coordinates, inclusive heading-local cm for LAFAN/Dog and the human experiment;
    otherwise generic exclusive world-displacement metres. both: world+training.
    source/all additionally show the original AMASS Z-up or CMU Y-up/Z-forward
    frame, read independently before the adapter's world-axis conversion.
    """
    if space not in ("source", "world", "training", "both", "all") or start < 0 or frames < 2:
        raise ValueError("Expected source/world/training/both/all, start >= 0, and frames >= 2")
    if space in ("source", "all") and source not in ("amass", "cmu"):
        raise ValueError("Source-frame comparison currently supports AMASS/CMU; other adapters support world/training/both")
    if source in ("amass", "cmu"):
        adapter = human_adapter(source, body_models)
        clip = adapter._read_clip(path, dtype=torch.float64)
    else:
        if body_models is not None:
            raise ValueError("body_models only applies to AMASS")
        adapter = build_adapter(source)
        if source == "bandai":
            import bvhio
            data = bvhio.readAsBvh(str(path))
            joints, parents = flatten_hierarchy(data.Root.Children[0])
            adapter._topology = SkeletonTopology(torch.tensor(parents), [joint.Name for joint in joints], {})
        clip = adapter._read_clip(path)
    if clip is None:
        raise ValueError(f"{path}: empty clip")
    indices = resample_indices(len(clip.joint_rotations), clip.fps, 30.)
    needed = frames + (space in ("training", "both", "all"))
    if start + needed > len(indices):
        raise ValueError(f"{path}: need {needed} sampled frames from {start}; only {len(indices)} at 30 Hz")
    indices = indices[start:start+needed]
    q, root = clip.joint_rotations[indices], clip.root_position[indices]
    parents = adapter._topology.parents
    panels = []
    common = {"fps": 30., "path": path, "domain": source}
    if space in ("source", "all"):
        if source == "amass":
            with np.load(path) as data:
                original_q = axis_angle_to_quaternion(torch.from_numpy(data["poses"][indices[:frames]]).double().reshape(frames, -1, 3))[:, :22]
                bm = adapter._body_model(np.asarray(data["gender"]).item())
                rest = bm(betas=torch.from_numpy(data["betas"]).float()[None]).Jtr[0, :22].double()
                original_root = rest[0] + torch.from_numpy(data["trans"][indices[:frames]]).double()
            original_offsets = torch.zeros_like(rest)
            original_offsets[1:] = rest[1:] - rest[parents[1:]]
            up, role = "Z", "source: AMASS Z-up, before world-axis conversion"
        else:
            asf = adapter._asf_model(path)
            raw_root, _, raw_q = amc_to_tensors(asf, parse_amc(path))
            original_root = torch.from_numpy(raw_root)[indices[:frames]]
            original_q = torch.from_numpy(raw_q)[indices[:frames]]
            original_offsets = torch.from_numpy(asf.offsets)
            up, role = "Y", "source: CMU Y-up/Z-forward, ASF units converted to m"
        original, _ = ForwardKinematics(parents, original_offsets.expand(frames, -1, -1))(original_q, original_root)
        panels.append({**common, "positions": original.numpy(), "position_parents": parents.numpy(),
                       "units": "meters", "up_axis": up, "role": role})
    if space in ("world", "both", "all"):
        positions, _ = ForwardKinematics(parents, clip.offsets.expand(frames, -1, -1))(q[:frames], root[:frames])
        panels.append({**common, "positions": positions.numpy(), "position_parents": parents.numpy(),
                       "units": "meters", "role": "adapter world: Y-up, absolute trajectory"})
    if space in ("training", "both", "all"):
        if source in ("amass", "cmu"):
            offsets = clip.offsets * 100
            channels, _ = preprocess_reference_window(q, root * 100, offsets, parents, "lafan1",
                                                       forward_axis=human_forward_axis(source, offsets), stable_heading=True)
            channels = human_rotation_channels(source, channels)
            asf = adapter._asf_model(path) if source == "cmu" else None
            offsets = human_fk_offsets(source, clip.offsets, asf) * 100
            fk = JointForwardKinematics(HUMAN_FK_PARENTS[source], inclusive_velocity=True, not_end=HUMAN_ROTATION_IDS[source])
            units, role = "centimeters", "EXPERIMENTAL human LAFAN-style: heading-local/inclusive"
        elif source in ("lafan1", "dog"):
            raw = read_lafan_reference_bvh(path)
            expected = REFERENCE_HIERARCHIES[source]
            actual = (raw.parents.tolist(), raw.parents_withend.tolist(), raw.not_end.tolist())
            if actual != expected:
                raise ValueError("Unexpected reference joint/End Site mapping")
            step = 2 if source == "dog" else 1
            selection = slice(start*step, (start+frames+1)*step, step)
            channels, _ = preprocess_reference_window(raw.rotations[selection], raw.root_position[selection],
                                                       raw.offsets, raw.parents, source)
            offsets = raw.offsets_withend
            fk = JointForwardKinematics(raw.parents_withend, inclusive_velocity=True, not_end=raw.not_end.tolist())
            units, role = "centimeters", "reference training: heading-local/inclusive, End Sites"
        else:
            channels = to_joint_channels(q, root)
            offsets, fk = clip.offsets, JointForwardKinematics(parents)
            units, role = "meters", "generic PAN: world displacement/exclusive (NOT Mixamo reference)"
        positions = fk.world_positions(fk.bone_vectors_with_absolute_root(channels[None], offsets[None]))[0]
        panels.append({**common, "positions": positions.numpy(), "position_parents": fk.parents,
                       "units": units, "role": role})
    return panels

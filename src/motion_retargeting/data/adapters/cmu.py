"""
Adapter for the CMU Graphics Lab motion capture database, stored as ASF/AMC
files (one ASF skeleton per subject, many AMC motions per subject).

Every CMU subject shares the same joint topology but has its own bone lengths,
so each clip's MotionData carries its subject's offsets.
"""

from __future__ import annotations

from .base import MotionSourceAdapter

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple
import numpy as np
import torch

from motion_retargeting.data.representations.motion import MotionData
from motion_retargeting.kinematics.quaternions import quat_mul


INCHES_TO_METERS = 0.0254

# Bones below the wrists are dropped
EXCLUDED_BONES = ("lhand", "lthumb", "lfingers", "rhand", "rthumb", "rfingers")

# no toes_end joints on either foot.
NO_END_SITE_BONES = ("ltoes", "rtoes")


class CMUAdapter(MotionSourceAdapter):
    cache_version = "metres-v2"

    def __init__(
        self,
        device: str = "cpu",
    ):
        super().__init__("cmu", device=device)

        self._asfs = {}

    def _extract(self) -> List[MotionData]:
        files = sorted(p for p in self.raw_dir.rglob(f"*.amc"))

        from tqdm import tqdm

        motions: List[MotionData] = []
        for path in tqdm(files, desc="Extracting CMU Data"):
            motion = self._read_clip(path)
            if motion is not None:
                motions.append(motion)
        return motions

    def _read_clip(self, path: Path, dtype: torch.dtype = torch.float32) -> MotionData | None:
        """Read one ASF/AMC pair without touching caches or requiring a corpus conversion."""
        asf = self._asf_model(path)
        frames = parse_amc(path)
        if not frames:
            return None
        root_positions, _, local_rotations = amc_to_tensors(asf, frames)
        root_positions_yup = zforward_to_xforward_vector(torch.as_tensor(root_positions))
        local_rotations[:, 0] = zforward_to_xforward_quaternion(torch.as_tensor(local_rotations[:, 0]))
        return self._construct_motion(
            parents=asf.parents, offsets=asf.offsets, joint_rotations=local_rotations,
            root_position=root_positions_yup, fps=120.0, joint_names=asf.joint_names, dtype=dtype,
        )

    def _asf_model(self, amc_file: Path) -> "_ASF":
        """ The parsed ASF skeleton of an AMC file's subject ("01_02.amc" -> "01.asf"), parsed once. """
        subject = amc_file.stem.split("_")[0]
        if subject not in self._asfs:
            self._asfs[subject] = parse_asf(amc_file.with_name(f"{subject}.asf"))
        return self._asfs[subject]


class _ASF:
    """
    Parsed ASF skeleton, flattened to joints.
    """

    def __init__(self):
        self.scale: float = 1.0

        self.root_order: List[str] = []
        self.root_axis: Tuple[List[float], str] = ([0.0, 0.0, 0.0], "XYZ")

        self.bone_vectors: Dict[str, np.ndarray] = {}
        self.bone_axes: Dict[str, Tuple[List[float], str]] = {}
        self.bone_dofs: Dict[str, List[str]] = {}
        self.children: Dict[str, List[str]] = {}

        self.joint_names: List[str] = []
        self.parents: List[int] = []
        self.offsets: List[np.ndarray] = []
        self.joint_bone: List[str | None] = []  # bone whose rotation drives each joint (None for end sites)


def parse_asf(path: Path) -> _ASF:
    asf = _ASF()

    section = None
    bone: Dict[str, object] = {}
    length_unit = 1.0

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(":"):
            parts = line.split()
            section = parts[0][1:]
            continue

        tokens = line.split()

        if section == "units":
            if tokens[0] == "length":
                length_unit = float(tokens[1])

        elif section == "root":
            if tokens[0] == "order":
                asf.root_order = [t.lower() for t in tokens[1:]]
            elif tokens[0] == "axis":
                asf.root_axis = ([0.0, 0.0, 0.0], tokens[1])
            elif tokens[0] == "orientation":
                asf.root_axis = ([float(v) for v in tokens[1:4]], asf.root_axis[1])

        elif section == "bonedata":
            if tokens[0] == "begin":
                bone = {"dof": []}
            elif tokens[0] == "end":
                name = bone["name"]
                if name in EXCLUDED_BONES:
                    continue
                asf.bone_vectors[name] = np.asarray(bone["direction"]) * bone["length"]
                asf.bone_axes[name] = bone["axis"]
                asf.bone_dofs[name] = bone["dof"]
            elif tokens[0] == "name":
                bone["name"] = tokens[1]
            elif tokens[0] == "direction":
                bone["direction"] = [float(v) for v in tokens[1:4]]
            elif tokens[0] == "length":
                bone["length"] = float(tokens[1])
            elif tokens[0] == "axis":
                bone["axis"] = ([float(v) for v in tokens[1:4]], tokens[4])
            elif tokens[0] == "dof":
                bone["dof"] = tokens[1:]

        elif section == "hierarchy":
            if tokens[0] in ("begin", "end") or tokens[0] in EXCLUDED_BONES:
                continue
            asf.children[tokens[0]] = [child for child in tokens[1:] if child not in EXCLUDED_BONES]

    asf.scale = (1.0 / length_unit) * INCHES_TO_METERS
    for name in asf.bone_vectors:
        asf.bone_vectors[name] = asf.bone_vectors[name] * asf.scale

    # Flatten the bone hierarchy into joints (depth-first, parents before children).
    def add_joint(name: str, parent: int, offset: np.ndarray, bone_name: str | None) -> int:
        asf.joint_names.append(name)
        asf.parents.append(parent)
        asf.offsets.append(offset)
        asf.joint_bone.append(bone_name)
        return len(asf.joint_names) - 1

    def visit(bone_name: str, parent_joint: int, offset: np.ndarray) -> None:
        joint = add_joint(bone_name, parent_joint, offset, bone_name)

        for child in asf.children.get(bone_name, []):
            visit(child, joint, asf.bone_vectors[bone_name])

        # Only leaves of the ASF hierarchy get an end site; bones that lost their children
        # to EXCLUDED_BONES (the wrists) still have a hierarchy entry and stay plain leaves,
        # as do NO_END_SITE_BONES (the toes).
        if bone_name not in asf.children and bone_name not in NO_END_SITE_BONES:
            add_joint(f"{bone_name}_end", joint, asf.bone_vectors[bone_name], None)

    root = add_joint("root", -1, np.zeros(3), "root")
    for child in asf.children.get("root", []):
        if asf.bone_dofs.get(child):
            # Root bones start at the root, so their joints sit at the root (zero offset).
            visit(child, root, np.zeros(3))
        else:
            # A DOF-less root bone (lhipjoint / rhipjoint) never rotates, so its joint would
            # only duplicate the root: skip it and attach its children to the root directly,
            # offset by the bone's vector.
            for grandchild in asf.children.get(child, []):
                visit(grandchild, root, asf.bone_vectors[child])

    asf.offsets = np.stack(asf.offsets)

    return asf


def parse_amc(path: Path) -> List[Dict[str, List[float]]]:
    """ Returns one {bone_name: dof values} dict per frame. """
    frames: List[Dict[str, List[float]]] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue

        tokens = line.split()
        if len(tokens) == 1 and tokens[0].isdigit():
            frames.append({})
            continue

        if tokens[0] in EXCLUDED_BONES:
            continue

        frames[-1][tokens[0]] = [float(v) for v in tokens[1:]]

    return frames


def axis_rotation(axis: Tuple[List[float], str]) -> R:
    """ ASF `axis` (degrees + order, e.g. "XYZ") -> the bone's local-to-global frame rotation C. """
    angles, order = axis
    values = {c: a for c, a in zip("xyz", angles)}
    order = order.lower()
    return R.from_euler(order, [values[c] for c in order], degrees=True)


def amc_to_tensors(asf: _ASF, frames: List[Dict[str, List[float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      root_positions:  [T, 3] (meters), from the AMC root translation dofs
      root_rotations:  [T, 4] (w, x, y, z), from the AMC root rotation dofs
      local_rotations: [T, J, 4] (w, x, y, z), each C * M * C^-1 for its bone
    """
    num_frames = len(frames)
    num_joints = len(asf.joint_names)

    local_rotations = np.zeros((num_frames, num_joints, 4))
    local_rotations[..., 0] = 1.0  # identity for end sites and dof-less bones

    # Root: order e.g. TX TY TZ RX RY RZ.
    root_values = np.asarray([frame["root"] for frame in frames])
    translation_idx = [asf.root_order.index(c) for c in ("tx", "ty", "tz")]
    rotation_dofs = [dof for dof in asf.root_order if dof.startswith("r")]
    rotation_idx = [asf.root_order.index(dof) for dof in rotation_dofs]

    root_positions = root_values[:, translation_idx] * asf.scale

    bone_dofs = dict(asf.bone_dofs)
    bone_dofs["root"] = rotation_dofs
    axes = dict(asf.bone_axes)
    axes["root"] = asf.root_axis

    for joint, bone_name in enumerate(asf.joint_bone):
        if bone_name is None or not bone_dofs.get(bone_name):
            continue

        dofs = bone_dofs[bone_name]
        if bone_name == "root":
            angles = root_values[:, rotation_idx]
        else:
            angles = np.asarray([frame.get(bone_name, [0.0] * len(dofs)) for frame in frames])

        # Extrinsic rotation applied in the listed dof order (e.g. "rx ry rz" -> Rz Ry Rx).
        motion = R.from_euler("".join(dof[1] for dof in dofs), angles, degrees=True)
        frame = axis_rotation(axes[bone_name])

        quat = (frame * motion * frame.inv()).as_quat()  # (x, y, z, w)
        local_rotations[:, joint] = quat[:, [3, 0, 1, 2]]

    # The ASF root bone is the pelvis; its rotation comes from the AMC root line.
    root_rotations = local_rotations[:, asf.joint_bone.index("root")].copy()

    return root_positions, root_rotations, local_rotations


def zforward_to_xforward_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    Physically rotate an orientation +90 degrees around the Y-axis.

    Quaternion convention: (w, x, y, z)
    """
    import math

    q_convert = torch.tensor(
        [
            math.cos(math.pi / 4),  # w
            0.0,                    # x
            math.sin(math.pi / 4),  # y
            0.0,                    # z
        ],
        dtype=q.dtype,
        device=q.device,
    )

    return quat_mul(q_convert, q)


def zforward_to_xforward_vector(v: torch.Tensor) -> torch.Tensor:
    """
    Convert vectors from:
        Y-up, Z-forward
    to:
        Y-up, X-forward

    v: [..., 3]
    """
    x, y, z = v.unbind(dim=-1)

    return torch.stack([
        z,   # new x
        y,   # new y
        -x,  # new z
    ], dim=-1)

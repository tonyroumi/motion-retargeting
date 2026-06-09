"""
Rollout motion clips through MuJoCo to extract kinematic state for AMP,
including ctrl values for position-servo actuator validation.

For each .npz clip in the input directory, sub-steps between consecutive
keyframes using mj_integratePos and saves the resulting kinematic quantities
at each sub-step. Output spacing is set to the *control* timestep used during
training so that discriminator reference transitions are sampled at the same
rate as policy transitions.

Between keyframes i and i+1 (separated by frame_dt = 1/fps):
  - velocity is computed once via mj_differentiatePos
  - mj_integratePos propagates qpos forward by control_dt each step
  - each intermediate state is recorded

The control timestep must be an integer divisor of frame_dt; this is asserted.

Output length: (N - 1) * num_substeps + 1  where
  num_substeps = frame_dt / control_dt   (must be integer)
  control_dt   = decimation / sim_freq

Position-servo ctrl convention
-------------------------------
For a position-servo actuator, ctrl[i] is the *target joint angle* sent to
the PD controller.  During a kinematic mocap replay the "correct" target is
the joint angle that produces the observed pose, i.e.

    ctrl[i] = qpos[7 + i]

This script records exactly that so you can compare it against your policy's
output and confirm the policy is operating in the correct space.

If your model has fewer actuators than DOFs (nu < nq - 7), only the first
model.nu joints are mapped; the script asserts nu <= nq - 7 to catch
mis-configured models early.

Saved per-clip (float32 npz):
  root_pos      (M, 3)          world-frame root position
  root_rot      (M, 4)          root quaternion (w, x, y, z)
  root_vel      (M, 3)          root linear velocity (world frame)
  root_ang_vel  (M, 3)          root angular velocity (body-local frame)
  dof_pos       (M, nq-7)       joint positions (qpos minus free-root slots)
  body_pos      (M, nbody, 3)   world-frame body positions (full xpos, index 0 is world)
  joint_pos     (M, njoint, 3)  world-frame joint anchor positions (xanchor)
  dof_vel       (M, nv-6)       joint velocities (qvel minus free-root slots)
  ctrl          (M, nu)         position-servo targets = qpos[7 : 7+nu]

Velocity convention note: for a free root joint, qvel[0:3] is linear velocity
in the world frame and qvel[3:6] is angular velocity in the body-local frame.
Discriminator observations built from the live sim must use the same
convention or there will be a silent distribution mismatch.

Usage:
    # Training runs at sim_freq=200, decimation=4 (control at 50 Hz):
    python rollout_motions.py \\
        --output data/humanoid_rollout \\
        --sim-freq 200 --decimation 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def find_xml(directory: Path) -> Path:
    xmls = sorted(directory.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No .xml file found in {directory}")
    if len(xmls) > 1:
        print(f"[warn] multiple XML files found, using {xmls[0].name}")
    return xmls[0]


# ---------------------------------------------------------------------------
# State recording
# ---------------------------------------------------------------------------

def _record(
    data: mujoco.MjData,
    out: dict,
    idx: int,
    nu: int,
) -> None:
    """Write all kinematic quantities + position-servo ctrl into output arrays.

    ctrl is recorded as qpos[7 : 7+nu].  For a position servo the controller
    target that would reproduce the mocap pose is exactly the current joint
    angle, making this the ground-truth ctrl the policy should be imitating.
    """
    out["root_pos"][idx]     = data.qpos[:3]
    out["root_rot"][idx]     = data.qpos[3:7]
    out["root_vel"][idx]     = data.qvel[:3]
    out["root_ang_vel"][idx] = data.qvel[3:6]
    out["dof_pos"][idx]      = data.qpos[7:]
    out["body_pos"][idx]     = data.xpos
    out["joint_pos"][idx]    = data.xanchor
    out["dof_vel"][idx]      = data.qvel[6:]

    # Position-servo target: the joint angle the actuator should be commanded
    # to in order to reproduce this frame.  Slice to nu in case the model has
    # fewer actuators than total DOFs (e.g. the root is unactuated).
    out["ctrl"][idx] = data.qpos[7 : 7 + nu]


# ---------------------------------------------------------------------------
# Per-clip rollout
# ---------------------------------------------------------------------------

def rollout_clip(
    frames: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_dt: float,
    control_dt: float,
) -> tuple[dict, int]:
    """
    Kinematically replay frames with sub-step interpolation at the control rate.

    Between each pair of keyframes the inter-frame velocity is held constant
    and mj_integratePos steps qpos forward by control_dt, giving smooth
    intermediate states that respect the quaternion manifold.

    Requires control_dt to be an integer divisor of frame_dt.
    """
    N  = frames.shape[0]
    nv = model.nv
    nu = model.nu

    # Sanity-check: every actuator must map to a DOF slot in qpos.
    # For position servos this is always true unless the XML is misconfigured.
    n_joint_dofs = model.nq - 7
    if nu > n_joint_dofs:
        raise ValueError(
            f"model.nu ({nu}) > nq - 7 ({n_joint_dofs}). "
            "More actuators than joint DOFs — check your XML."
        )

    # Exact integer divisor required — control rate must align with mocap fps.
    ratio = frame_dt / control_dt
    num_substeps = round(ratio)
    if not np.isclose(ratio, num_substeps, rtol=1e-6):
        raise ValueError(
            f"control_dt ({control_dt}) must evenly divide frame_dt ({frame_dt}); "
            f"got ratio {ratio:.6f}. Pick a control frequency that is an "
            f"integer divisor of the mocap fps ({1.0 / frame_dt:.2f})."
        )
    num_substeps = max(1, num_substeps)
    total = (N - 1) * num_substeps + 1

    nbody  = model.nbody
    njoint = model.njnt

    out = dict(
        root_pos    = np.zeros((total, 3),           dtype=np.float32),
        root_rot    = np.zeros((total, 4),           dtype=np.float32),
        root_vel    = np.zeros((total, 3),           dtype=np.float32),
        root_ang_vel= np.zeros((total, 3),           dtype=np.float32),
        dof_pos     = np.zeros((total, n_joint_dofs),dtype=np.float32),
        body_pos    = np.zeros((total, nbody,  3),   dtype=np.float32),
        joint_pos   = np.zeros((total, njoint, 3),   dtype=np.float32),
        dof_vel     = np.zeros((total, nv - 6),      dtype=np.float32),
        ctrl        = np.zeros((total, nu),          dtype=np.float32),
    )

    qvel  = np.zeros(nv)
    q_sub = np.zeros(model.nq)
    out_idx = 0

    for i in range(N - 1):
        # Constant velocity for this inter-frame interval.
        mujoco.mj_differentiatePos(model, qvel, frame_dt, frames[i], frames[i + 1])

        for k in range(num_substeps):
            # Integrate from keyframe i by k * control_dt.
            q_sub[:] = frames[i]
            if k > 0:
                mujoco.mj_integratePos(model, q_sub, qvel, k * control_dt)

            data.qpos[:] = q_sub
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            _record(data, out, out_idx, nu)
            out_idx += 1

    # Terminal frame — qvel is held from the last inter-frame interval.
    # ctrl is still well-defined: the servo target for the final pose.
    data.qpos[:] = frames[-1]
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    _record(data, out, out_idx, nu)

    return out, num_substeps


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def process_directory(
    src_dir: Path,
    out_dir: Path,
    control_dt: float,
) -> None:
    xml_path = find_xml(src_dir)
    print(f"Loading model : {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    nu           = model.nu
    n_joint_dofs = model.nq - 7
    print(f"Actuators (nu): {nu}  |  Joint DOFs (nq-7): {n_joint_dofs}")
    if nu == 0:
        print("[warn] model.nu == 0 — no actuators defined. "
              "ctrl array will be empty. Is this the right XML?")

    clips = sorted(src_dir.glob("*.npz"))
    if not clips:
        print(f"No .npz files found in {src_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fps = 1.0 / control_dt
    print(f"Control dt    : {control_dt:.6f} s  ({out_fps:.2f} Hz)\n")

    for clip_path in clips:
        d      = np.load(str(clip_path), allow_pickle=False)
        name   = str(d["name"])
        fps    = float(d["fps"])
        frames = d["frames"].astype(np.float64)

        frame_dt = 1.0 / fps

        if frames.shape[1] != model.nq:
            print(
                f"[skip] {clip_path.name}: "
                f"frame width {frames.shape[1]} != nq {model.nq}"
            )
            continue

        data = mujoco.MjData(model)
        try:
            state, num_substeps = rollout_clip(
                frames, model, data, frame_dt, control_dt
            )
        except ValueError as e:
            print(f"[skip] {clip_path.name}: {e}")
            continue

        out_frames = state["root_pos"].shape[0]
        out_path   = out_dir / clip_path.name
        np.savez_compressed(
            str(out_path),
            name = np.array(name),
            fps  = np.float32(out_fps),
            dt   = np.float32(control_dt),
            nu   = np.int32(nu),
            **state,
        )
        print(
            f"  {clip_path.name}: "
            f"{frames.shape[0]} keyframes × {num_substeps} substeps"
            f" = {out_frames} frames @ {out_fps:.1f} Hz"
            f"  |  ctrl shape {state['ctrl'].shape}"
            f"  →  {out_path}"
        )

    # Quick validation summary printed to stdout so you can eyeball ranges.
    _print_ctrl_summary(out_dir)


# ---------------------------------------------------------------------------
# Validation helper — print ctrl range across all saved clips
# ---------------------------------------------------------------------------

def _print_ctrl_summary(out_dir: Path) -> None:
    """Load every saved clip and report per-DOF ctrl min/max.

    This is the fast sanity check: if any column is identically zero or
    wildly out of the expected joint-limit range, the actuator mapping is
    likely wrong.
    """
    clips = sorted(out_dir.glob("*.npz"))
    if not clips:
        return

    all_ctrl: list[np.ndarray] = []
    for p in clips:
        d = np.load(str(p), allow_pickle=False)
        if "ctrl" in d:
            all_ctrl.append(d["ctrl"])

    if not all_ctrl:
        print("\n[warn] No ctrl arrays found in output files.")
        return

    ctrl = np.concatenate(all_ctrl, axis=0)  # (total_frames, nu)
    nu   = ctrl.shape[1]

    print("\n--- ctrl validation summary (position-servo targets, radians) ---")
    print(f"{'DOF':>5}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
    print("-" * 52)
    for j in range(nu):
        col = ctrl[:, j]
        print(
            f"{j:>5}  {col.min():>10.4f}  {col.max():>10.4f}"
            f"  {col.mean():>10.4f}  {col.std():>10.4f}"
        )
    print("-" * 52)
    print(
        f"  Total frames: {ctrl.shape[0]}  |  "
        f"Global range: [{ctrl.min():.4f}, {ctrl.max():.4f}]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: <src_dir>_rollout)",
    )
    parser.add_argument(
        "--sim-freq",
        type=float,
        required=True,
        help="MuJoCo simulation frequency used during training (Hz)",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=1,
        help=(
            "Sim steps per control step during training "
            "(control_freq = sim_freq / decimation)"
        ),
    )
    args = parser.parse_args()

    src_dir = Path("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/humanoid")
    if not src_dir.is_dir():
        print(f"Error: {src_dir} is not a directory")
        sys.exit(1)

    if args.decimation < 1:
        print(f"Error: --decimation must be >= 1, got {args.decimation}")
        sys.exit(1)

    control_dt = args.decimation / args.sim_freq

    out_dir: Path = (
        args.output.resolve() if args.output
        else src_dir.parent / (src_dir.name + "_rollout")
    )

    process_directory(src_dir, out_dir, control_dt)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
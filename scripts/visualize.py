#!/usr/bin/env python
"""Side-by-side dataset-coordinate and retarget-NPZ inspection; no Isaac Lab.

Examples:
  python scripts/visualize.py outputs outputs/retarget --save-dir /tmp/retarget-viz --frame 0
  python scripts/visualize.py datasets --clip amass=/data/02_01_poses.npz --clip cmu=/data/02_01.amc \
      --body-models /data/body_models --space both --save-dir /tmp/dataset-viz --frame 0

Adapted from the standalone visualize_retarget.py supplied with the task. Output
NPZs follow https://github.com/tonyroumi/motion-retargeting/pull/29. All panels are
shown in metres on a shared physical scale; Y-up normally, or explicit source
Z-up for raw AMASS. Input cm/m and coordinate stage
remain labeled; no normalization or source-heading restoration is invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


UNIT_SCALE = {"meters": 1., "metres": 1., "m": 1., "centimeters": .01, "centimetres": .01, "cm": .01}


def validated_motion(motion: dict, units_override: str | None = None) -> dict:
    result = dict(motion)
    positions = np.asarray(motion["positions"], dtype=np.float64)
    parents = np.asarray(motion["position_parents"])
    if positions.ndim != 3 or positions.shape[-1] != 3 or min(positions.shape[:2]) < 1 or not np.isfinite(positions).all():
        raise ValueError("positions must be finite, nonempty [T,J,3]")
    if parents.shape != (positions.shape[1],) or parents.dtype.kind not in "iu" or parents[0] != -1:
        raise ValueError("position_parents must be integer [J] with root -1")
    if any(not 0 <= int(parent) < joint for joint, parent in enumerate(parents[1:], 1)):
        raise ValueError("position_parents must be parent-before-child")
    units = str(units_override or motion.get("units", motion.get("checkpoint_physical_units", "")))
    if units not in UNIT_SCALE:
        raise ValueError("Missing/unknown physical units; supply --units meters or centimeters for legacy NPZs")
    fps = float(motion.get("fps", 30.))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Input fps must be finite and positive")
    up = str(motion.get("up_axis", "Y"))
    if up not in ("Y", "Z"):
        raise ValueError("Only explicit Y-up or Z-up frames are supported")
    result.update(positions=positions * UNIT_SCALE[units], position_parents=parents, fps=fps, input_units=units, up_axis=up)
    return result


def load_output(path: Path, units_override: str | None = None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    motion["path"] = path
    return validated_motion(motion, units_override)


def find_groups(inputs: list[str]) -> list[tuple[str, list[Path]]]:
    groups, loose = [], []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            loose.append(path)
        elif path.is_dir():
            directories = [path] if list(path.glob("*.npz")) else sorted(p for p in path.iterdir() if p.is_dir() and list(p.glob("*.npz")))
            if not directories:
                raise ValueError(f"No NPZs in {path} or its immediate subdirectories")
            groups.extend((directory.name, sorted(directory.glob("*.npz"))) for directory in directories)
        else:
            raise FileNotFoundError(path)
    if loose:
        groups.append(("outputs", loose))
    return groups


def make_figure(title: str, motions: list[dict], fixed_camera: bool = False, fps: float = 30.):
    """Render validated metre panels with their labeled up axis, sampled by time."""
    import matplotlib.pyplot as plt

    if not motions or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Need motions and a positive render fps")
    duration = min((len(m["positions"])-1)/m["fps"] for m in motions)
    num_frames = int(np.floor(duration*fps + 1e-9)) + 1
    times = np.arange(num_frames)/fps
    poses = [m["positions"][np.floor(times*m["fps"]+.5).astype(int).clip(0, len(m["positions"])-1)][
        ..., [0, 2, 1] if m["up_axis"] == "Y" else [0, 1, 2]] for m in motions]
    # Shared scale exposes unit/size errors rather than auto-fitting each body to
    # the same visual height. Matplotlib vertical Z displays the data's Y axis.
    radius = max(np.abs(p-p[:, :1]).max() for p in poses)*1.1 + 1e-6
    fixed_radius = max(np.ptp(p.reshape(-1, 3), axis=0).max()/2 for p in poses)*1.05 + 1e-6
    fig = plt.figure(figsize=(6*len(motions), 6.5))
    panels = []
    for i, (motion, positions) in enumerate(zip(motions, poses)):
        ax = fig.add_subplot(1, len(motions), i+1, projection="3d")
        pairs = [(int(parent), joint) for joint, parent in enumerate(motion["position_parents"]) if parent >= 0]
        color = "tab:orange" if str(motion.get("role")) == "retargeted" else "tab:blue"
        lines = [ax.plot([], [], [], color=color, linewidth=2)[0] for _ in pairs]
        dots = ax.plot([], [], [], "o", color="black", markersize=2)[0]
        root = positions[:, 0]
        track = np.column_stack((root[:, :2], np.full(num_frames, positions[..., 2].min())))
        track_line = ax.plot(*track.T, color="gray", alpha=.35)[0]
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)" if motion["up_axis"] == "Y" else "Y (m)")
        ax.set_zlabel(f"{motion['up_axis']} up (m)")
        role = motion.get("role", Path(motion["path"]).stem)
        ax.set_title(f"{motion.get('domain', '?')}: {role}\ninput {motion['input_units']} → displayed m; {positions.shape[1]} joints", fontsize=9)
        midpoint = (positions.reshape(-1, 3).min(0)+positions.reshape(-1, 3).max(0))/2
        panels.append((ax, positions, pairs, lines, dots, track, track_line, midpoint))

    def update(frame):
        if not 0 <= frame < num_frames:
            raise IndexError(f"Frame {frame} outside {num_frames} synchronized frames")
        artists = []
        for ax, positions, pairs, lines, dots, track, track_line, midpoint in panels:
            pose = positions[frame]
            for line, pair in zip(lines, pairs):
                line.set_data_3d(*pose[list(pair)].T)
            dots.set_data_3d(*pose.T)
            center, half = (midpoint, fixed_radius) if fixed_camera else (pose[0], radius)
            ax.set_xlim(center[0]-half, center[0]+half)
            ax.set_ylim(center[1]-half, center[1]+half)
            ax.set_zlim(center[2]-half, center[2]+half)
            visible = np.abs(track[:, :2]-center[:2]).max(1) <= half
            track_line.set_data_3d(*np.where(visible[:, None], track, np.nan).T)
            artists.extend([*lines, dots, track_line])
        fig.suptitle(f"{title} — labeled coordinate axes, common metre scale — {frame+1}/{num_frames}, t={frame/fps:.3f}s")
        return artists

    return fig, update, num_frames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    outputs = sub.add_parser("outputs", help="PR export NPZs, one figure per direction")
    outputs.add_argument("inputs", nargs="+")
    outputs.add_argument("--units", choices=("meters", "centimeters"), help="Explicit override for legacy NPZ metadata")
    datasets = sub.add_parser("datasets", help="Bounded raw clips through adapters, no caches or training")
    datasets.add_argument("--clip", action="append", required=True, metavar="ADAPTER=PATH")
    datasets.add_argument("--body-models", help="SMPL model parent directory for AMASS only")
    datasets.add_argument("--space", choices=("source", "world", "training", "both", "all"), default="world",
                          help="both = world+training; all adds independent source-frame FK (AMASS/CMU only)")
    datasets.add_argument("--start", type=int, default=0, help="Start frame on the sampled 30 Hz timeline")
    datasets.add_argument("--frames", type=int, default=64)
    for command in (outputs, datasets):
        command.add_argument("--save-dir", help="Write new files here instead of opening an interactive window")
        command.add_argument("--frame", type=int, help="One synchronized frame, PNG instead of animation")
        command.add_argument("--format", choices=("gif", "mp4"), default="gif")
        command.add_argument("--fps", type=float, default=30., help="Render timeline rate; input rates are respected")
        command.add_argument("--fixed-camera", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if args.save_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers

    if args.mode == "outputs":
        groups = [(title, [load_output(path, args.units) for path in paths]) for title, paths in find_groups(args.inputs)]
    else:
        from motion_retargeting.data.visualization.datasets import dataset_panels
        motions = []
        for item in args.clip:
            if "=" not in item:
                raise ValueError("--clip must be ADAPTER=PATH")
            source, path = item.split("=", 1)
            motions.extend(validated_motion(motion) for motion in dataset_panels(
                source, Path(path).expanduser(), args.space, args.start, args.frames,
                args.body_models if source == "amass" else None,
            ))
        groups = [("datasets", motions)]
    directory = Path(args.save_dir).expanduser() if args.save_dir else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    if directory and args.frame is None and args.format == "mp4" and not writers.is_available("ffmpeg"):
        raise ValueError("MP4 needs ffmpeg; use --format gif")
    animations = []
    for title, motions in groups:
        fig, update, count = make_figure(title, motions, args.fixed_camera, args.fps)
        suffix = f"_frame{args.frame:04d}.png" if args.frame is not None else f".{args.format}"
        output = directory / (title+suffix) if directory else None
        if output and output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        if args.frame is not None:
            update(args.frame)
            if output:
                fig.savefig(output, dpi=120)
        else:
            animation = FuncAnimation(fig, update, frames=count, interval=1000/args.fps)
            if output:
                animation.save(str(output), writer="pillow" if args.format == "gif" else "ffmpeg", fps=args.fps, dpi=80)
            else:
                animations.append(animation)
        if output:
            print(f"Saved {output}")
            plt.close(fig)
    if not directory:
        plt.show()


if __name__ == "__main__":
    main()

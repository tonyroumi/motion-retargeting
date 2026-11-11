# Side-by-side coordinate and retarget visualization

`scripts/visualize.py` combines bounded dataset inspection with the standalone
retarget-NPZ viewer workflow. It does not launch Isaac Lab, fit statistics, load an
entire corpus, update a model, or read/write processed motion caches.

## Dataset clips through the adapters

```sh
# Two converted datasets, side by side. These are real absolute-world poses,
# not normalized network channels or heading-removed training trajectories.
python scripts/visualize.py datasets \
  --clip amass=/data/amass/CMU/02/02_01_poses.npz \
  --clip cmu=/data/cmu/02/02_01.amc \
  --body-models /data/assets/body_models --space world

# For one human clip: original source frame | adapter world | training coordinates.
# AMASS original Z-up and adapter Y-up are labeled separately, not silently conflated.
python scripts/visualize.py datasets \
  --clip amass=/data/amass/CMU/02/02_01_poses.npz \
  --body-models /data/assets/body_models --space all \
  --save-dir /tmp/amass-coordinates --frame 0

# Adapter world and reference training coordinates, including End Sites in training FK:
python scripts/visualize.py datasets --clip lafan1=/data/LAFAN1/run1_subject2.bvh \
  --space both --start 1000 --frames 64 --save-dir /tmp/lafan-coordinates
```

`--clip ADAPTER=PATH` can repeat; registered names are `amass`, `cmu`, `bandai`,
`lafan1`, `dog`, `mixamo_a`, `mixamo_b`. BVH group-B Mixamo still requires the
already-split `_m` topology. No label-clicking UI is opened. Readers inspect one
whole input clip, then display the requested bounded range; AMASS needs its SMPL-H
assets, CMU needs the sibling `<subject>.asf`. The preview does not synthesize
missing raw data, assets, joints or Mixamo preprocessing equivalence.

- `--space world` (default): generic adapter FK, absolute trajectories in metres,
  Y-up. This is suitable for checking physical size/root coordinates before PAN.
- `--space training`: **denormalized** PAN training coordinates. LAFAN1/Dog use
  original BVH cm, heading removal, local velocity/inclusive integration and End
  Sites. CMU/AMASS use the explicitly experimental human LAFAN-style protocol,
  including real ASF endpoints; these panels say **EXPERIMENTAL**. BANDAI/Mixamo
  show generic world-displacement/exclusive integration in metres, labeled as
  **not** the Mixamo reference representation.
- `--space both`: world plus training for every supplied clip.
- `--space source` / `all`: independent pre-world-axis-conversion source FK, or
  source+world+training. Currently these source-frame comparisons support
  AMASS/CMU only. AMASS source is Z-up; CMU source is Y-up/Z-forward with ASF lengths
  already converted to metres. Other adapters still support world/training/both.

`--start` and `--frames` refer to the **30 Hz sampled timeline**, not native source
frame indices; default is start0/64 frames. Training previews require one extra
lookahead frame, and reproduce within-window yaw and dog smoothing. They use the
selected clip's own skeleton, not a corpus's first-per-subject bank choice. There
is no network normalization to visualize as spatial coordinates: these are FK
poses before normalization or after exact denormalization.

## Retargeted output NPZs

Compatible with the export schema in
https://github.com/tonyroumi/motion-retargeting/pull/29:
`positions [T,J_fk,3]`, `position_parents [J_fk]`, `fps`,
`checkpoint_physical_units`, and optional role/domain/bank metadata.

```sh
# One original|retargeted figure per discovered direction directory:
python scripts/visualize.py outputs outputs/retarget --save-dir /tmp/retarget-viz

# Explicit NPZs side by side, or one still frame:
python scripts/visualize.py outputs original.npz retargeted.npz \
  --save-dir /tmp/retarget-frame --frame 0
```

Archives load with `allow_pickle=False`. Invalid shapes, nonfinite coordinates,
invalid hierarchies, nonpositive/unknown frame rates and missing physical units
are rejected. Legacy files without unit metadata need an explicit
`--units meters` or `--units centimeters`; unit scale is never guessed.

## Rendering conventions and safety

All panels use **metres on one shared physical scale**; titles retain the input
cm/m label. Independent auto-scaling could hide a 100× adapter unit bug, so it is
not used. Axes are `X (m)`, `Z (m)`, `Y up (m)` for canonical data, or explicit
`X (m)`, `Y (m)`, `Z up (m)` for AMASS source coordinates. Source-axis conversion
can change apparent orientation between panels; the labels specify which axes
are being viewed. No root-height correction, heading restoration, IK cleanup or
skeleton-size normalization is silently applied.

Default camera follows each root with the same physical extent. `--fixed-camera`
frames complete trajectories, still at shared scale. Playback uses the common
duration with nearest-time frame selection, respecting each input's frame rate;
`--fps` chooses the rendered timeline rate (default30), not a mislabeled input rate.
A faint root track is projected to each panel's lowest observed joint height for
orientation only, not an inferred physical ground plane/contact annotation.

Without `--save-dir`, figures are interactive. With it, rendering is headless:
`--frame N` writes PNG; otherwise GIF (Pillow), or `--format mp4` with ffmpeg.
**Existing output files are never overwritten.** Use a new output directory for
another inspection. Inputs are always read-only.

Tests cover cm↔m scale, Y-up/Z-up labels, time synchronization, invalid metadata,
no-overwrite behavior, PNG/GIF generation, and the distinction between absolute
world and heading-local/inclusive training coordinates. Real CMU/AMASS prefix
plots and independent source→adapter FK comparisons were also checked. Plots are
coordinate diagnostics, not evidence of trained retargeting quality.

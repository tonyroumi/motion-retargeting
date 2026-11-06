# Training on the PAN authors' datasets

These sources load the datasets of the [PAN reference implementation](https://github.com/hlcdyy/pan-motion-retargeting)
into this repo's `MotionData`. PAN then builds its edge and absolute-root windows from them, and the shared
`MotionDatasetBuilder` serves SAN by removing heading, resampling and windowing them. The datasets are not downloaded by
this repo.

## Raw data

Paths are relative to the repo root. The `data/` directory is git-ignored.

| Source | Adapter | Expected directory | Skeleton |
|---|---|---|---|
| `mixamo_a` | `MixamoAdapter` | `data/raw/mixamo/{Aj,BigVegas,Kaya,SportyGranny}/*.bvh` | 23 joints |
| `mixamo_b` | `MixamoGroupBAdapter` | `data/raw/mixamo/{Malcolm_m,Remy_m,...}/*.bvh` (the 20 `GROUP_B_CHARACTERS` in `mixamo.py`) | 28 joints |
| `lafan1` | `LAFAN1Adapter` | `data/raw/lafan1/*.bvh` and `data/raw/lafan1/train.txt` | 22 joints |
| `dog` | `DogAdapter` | `data/raw/dog/*.bvh` and `data/raw/dog/train.txt` | 21 joints |

- **Train split.** `train.txt` selects the reference's training clips. Copy the reference's
  `data_preprocess/Lafan1_and_dog/lafan1_train.txt` and `dog_train.txt` to these names. Without it, the adapter warns and
  loads every BVH, including the reference's test clips.
- **Mixamo group B.** The `_m` directories are the output of the reference's `data_preprocess/Mixamo/split_joint.py`,
  which inserts the `*_split` joints. This repo does not generate them. At the reference's commit `5a51d92` that script's
  `import outer_utils as BVH` does not expose `BVH.load`, so it needs fixing before it runs.
- **Caches.** Adapters cache to `data/processed/<source>.pt`. Delete that file after changing the raw clips.

## Training

`configs/methods/pan.yaml` is the only PAN config. To train on Mixamo, point its groups at these sources:

```yaml
dataset:
  group_a:
    sources: [mixamo_a]
  group_b:
    sources: [mixamo_b]
```

and keep the rest of the file. PAN follows the official Mixamo path, so it needs the six-chain humanoid topology
(`humanoid_chains` in `methods/pan/masks.py`). `mixamo_a`, `mixamo_b` and `lafan1` have it. **The dog does not: its tail makes
a seventh chain, so PAN refuses that skeleton with an error.** The official LAFAN1/dog path is a different computation and is
not implemented here. The `lafan1` and `dog` loaders still load their clips, and both still go through the shared builder.

## What the adapters do

- Read rotations in each joint's own `CHANNELS` order, as wxyz quaternions.
- Convert centimeters to meters.
- Take the root position from its position channels, so the root `OFFSET` is not added twice.
- Drop End Sites and any joint the reference does not use.
- Re-root and collapse joints only when the result has constant offsets. Otherwise they raise an error.
- Assign body parts by name, so the interactive labeler does not open.

**LAFAN and dog root frames.** The builder reads heading from the root's local +X, and neither file gives it a forward
axis there. LAFAN authors its hips with +X up the spine and +Y forward. The dog has +Y up and +X pointing at its tail. Each
adapter moves a fixed half turn out of the root and into its children, which leaves world joint positions unchanged.
Afterwards both are +X forward, +Y up, +Z right, so both domains' canonical root velocity is +X when walking forward.

**Mixamo root frame.** Mixamo is assumed to be world-aligned at identity with the character facing +Z, like AMASS and CMU;
both Mixamo groups share that frame. This could not be checked without real Mixamo clips. The adapters warn if a clip's
root +X is mostly vertical.

## Differences from the reference, shared builder only

PAN's windows follow the reference's Mixamo computation; see the README for what remains different. These differences apply
only to windows from the shared `MotionDatasetBuilder`, which PAN does not use:

| | Reference | Shared builder |
|---|---|---|
| Rotation groups, Mixamo | One per edge, holding the parent joint's rotation | One per joint |
| Heading and root, Mixamo | Heading kept, absolute root position | Heading removed, velocity row |
| Heading axis, human | Root +Y, measured from +Z | The same axis after the LAFAN rebase, measured from +X: a constant 90 degree offset |
| Heading axis, dog | Root-to-`Spine1`, Gaussian-smoothed | Root's forward axis after the rebase, measured from +X, unsmoothed |
| Quaternion sign, LAFAN/dog | Forced to `w >= 0` | Not changed |
| Normalization | Every channel, per character (Mixamo) or per dataset (LAFAN/dog) | Velocity row only, per domain, from the train split |
| Augmentation, Mixamo | Time reversal with p = 0.5 | None |
| Downsampling | `::2` on the frames, dog and Mixamo | Builder's nearest-frame resampling from the file's rate to `target_fps` |
| Body parts, Mixamo | `Spine2` shared by four parts | Six disjoint parts with the same keys in both groups |
| Body parts, LAFAN/dog | Human arms, dog tail, and root in no part | Complete partition: arms join limbs, tail and root join the trunk |
| Offsets | Raw BVH units | Meters |

## Unverified against real data

No Mixamo clip is available here, so the Mixamo adapter is only tested on synthetic BVHs. Check these on real clips:

- **Frame rate.** The reference takes `::2` whatever the file's rate. `target_fps: 30` matches that only if the files say
  60 Hz. If they say 30 Hz, the reference's effective rate is 15 Hz and the config needs `target_fps: 15` to match.
- **Units.** Centimeters are assumed, as in LAFAN1 and the dog.
- **Group A hierarchy.** The reference re-roots type-0 files with more than 30 joints at joint 1, which its joint list implies is `Pelvis`.
  The adapter raises an error if re-rooting leaves an offset that moves, as it does if `Hips` is the file root and `Pelvis`
  rotates under it. The reference keeps a constant offset there and ignores that motion. If the real clips hit this error,
  group A cannot load until that approximation is added on purpose.

The dog demo's header says 30 Hz and `dog_std.bvh` says 60 Hz. The reference halves every dog clip regardless, and its
code comment says its dog data is 60 Hz, so `target_fps: 30` matches on the training clips if their headers say
60 Hz. The demo clip stays at 30 Hz here, where the reference's demo would reduce it to 15 Hz.

`skeleton_height` in `data/utils.py` measures the rest pose along an axis. It does not give a body height for LAFAN1 or the
dog, whose identity pose is not standing. PAN's training losses do not use it; they use `path_heights` in
`methods/pan/kinematics.py`.

# Opt-in LAFAN1/Dog reference semantics

Use `configs/methods/pan_lafan_dog_reference.yaml`, replacing both
`reference_archive.directory` values with the directory containing the **trusted**
reference `humtrain.npz`, `humtest.npz`, `humstats.npz`, `dogtrain.npz`,
`dogtest.npz`, and `dogstats.npz`. These archives contain pickled skeleton dictionaries;
do not use untrusted files. There is no runtime dependency on the reference source tree.

This is a bounded compatibility path, **not a full reproduction or a convergence
claim**. Exactly one `lafan1` domain and one `dog` domain are required; group names
can be arbitrary. Do not combine sources/params with a `reference_archive` or
`reference_raw` block. No raw-data directory renaming is necessary.

## Raw data and adapter selection

`configs/methods/pan_lafan_dog_raw.yaml` uses `sources: [lafan1]` and
`sources: [dog]`. PAN routes these adapters through the reference preprocessing
path, **not** the generic MotionCollection builder. It reads `data/raw/lafan1` and
`data/raw/dog`, falling back to existing `LAFAN1` / `Dog` folders. To read another
location without copying data, replace a group's `sources` with:

```yaml
reference_raw:
  kind: lafan1  # dog for the other domain
  directory: /path/to/LAFAN1
```

Raw loading uses the published train/test manifests included in
`data/adapters/lafan1_splits.py`: 17/6 human clips and 38/13 dog clips. Extra clips
are ignored; missing or duplicate stems fail. No arbitrary last-5% holdout, caps,
resampling overrides, or alternate windows are silently applied. The only
accepted `sources` params are the fixed `sequence_length: 64`, `stride: 20`, and
`target_fps: 30`; omit params entirely for the same behavior.

The precision-preserving BVH reader keeps float64 original-frame cm coordinates
and End Sites, verifies the reference names/topology across clips and 30/60 Hz
source rates, and selects dog frames 0,2,4,... . Non-root translation or differing
rotation orders are rejected, not approximated. Raw loading applies the same
per-window canonicalization and normalization policy described below. Statistics
are regenerated across **train plus test**, with population std and exact-zero-only
replacement, and shared by all skeleton banks. Subjects, not tiny per-clip offset
roundoff, define banks (5 human / 1 dog on the available corpus).

Raw enumeration is sorted; upstream used filesystem-dependent `os.listdir` order.
Both choose the first clip's skeleton per subject per split, so tiny within-subject
BVH offset differences can select a slightly different skeleton. In numerical
checks on the available corpus, mean/std errors were below 2.2e-13, 100 sampled
normalized windows differed by at most 2.4e-7, and skeleton differences were below
5.5e-5 cm. These are sampled input checks, not a training-convergence claim. Use
**reference_archive** for exact historical window order, skeleton choices and saved
statistics. The raw path never reads or writes MotionCollection caches; parsing
runs again on each load. Its regenerated-statistics fingerprint distinguishes its
checkpoints from archive checkpoints.

Outside PAN, generic LAFAN1/Dog adapters retain their rebased metre MotionCollection
contract and now select only the exact training manifests. Their versioned cache
names avoid reusing older all-clips caches.

## Data contract

- Explicit train/test sets (6566/2074 human, 2769/779 dog in the shipped
  reference), 64 frames at 30 Hz. No percent holdout or window cap on this path.
- Original BVH local frames and **centimetres**, not the generic adapters' rebased
  metre frames. The network retains 22/21 rotation joints (92/88 channels).
- Archive Q already has heading removed and V is heading-local displacement **per
  frame**, not velocity per second. Force quaternion `w >= 0`. Derive yaw increments
  inside each window from archived headings, repeating the last increment; do not
  use the next window/clip frame's yaw. Match the reference's FP32 yaw calculation.
- Normalize with the saved **domain-wide train-plus-test statistics**, including
  their test-distribution leakage, population std and tiny nonzero std values.
  Normalize in archive precision before conversion to FP32; never refit/floor these
  statistics. Training and validation share banks, including held-out skeletons.
- Exact three **partial** correspondence lists; human arms and dog tail remain
  unassigned. Root/velocity are shared, and unassigned decoder outputs read all parts.
  Do not reinterpret these lists as six complete anatomical parts.
- Expand identity End Site rotations for 27-joint FK using archived offsets and
  mapping. Integrate local root displacement **inclusively** and do not reintegrate
  yaw for training positions. Unused legacy height/five-end-effector construction
  is bypassed, not replaced with invented limbs.

`methods/pan/reference.py` also exposes `reference_window_starts` and
`preprocess_reference_window` for controlled raw-preprocessing comparisons. The
strict boundary is `start + W < frame_count - 1`; each preprocessing call takes
W poses plus one lookahead frame, already sampled at 30 Hz, in **original** BVH
frames/cm. Human heading is original-root +Y projected onto XZ; dog heading is
root-to-joint-2 direction, normalized, Gaussian-smoothed (sigma 8, nearest padding)
**within the window**, then normalized again. Both face canonical +Z. Heading is
removed from global rotations before converting back to local rotations; root
lookahead displacement is rotated by the inverse current heading. The raw loader
uses these helpers after precision-preserving BVH parsing, before any FP32 cast.

## Objective and routes

The example configuration implements the reference `norm_rec` objective, summed
over domains/routes:

```
selected normalized quaternion MSE
+ 100 * normalized root XYZ+yaw MSE
+ 0.01 * selected integrated-position MSE (cm²)
+ 0.001 * (selected cycle-position MSE (cm²) + cycle-latent MSE)
+ raw-score L2-GAN generator loss
+ 1000 * (mapped root-vector MSE + denormalized yaw-increment MSE)
```

Selections are root plus correspondence joints (14 human / 19 dog rotations),
plus the reference selected End Sites for position losses (17 / 23 positions).
Reconstruction lambda multiplies all three reconstruction terms. Position losses
have no root-relative conversion or height division. `root_velocity_lambda` is
reference-only. Use the example's `position_lambda: 0.01` and `cycle_lambda: 0.001`;
the legacy defaults intentionally remain unchanged.

Velocity mapping is `((norm(v)-min)/(max-min)) * v/norm(v)`, using the saved domain
speed extrema. It is not clamped to [0,1]. **Zero speed is undefined in the reference**:
this implementation retains its NaN behavior and the existing finite-loss guard
rejects such a training step. It does not silently choose a different epsilon or
stationary-motion objective. Degenerate heading directions likewise have no new
fallback invented here.

Exactly two routes run: human→dog and dog→human. Each uses the current destination
batch's skeleton IDs (equal batch sizes required), not random target banks; there
are no self-domain cycles. The discriminator consumes **all** 27 integrated joint
positions in cm, not selected joints, bones, normalized positions, or rotations.
It returns raw scores, with MSE targets 1/0 and the discriminator's usual factor
1/2. Replay is disabled on this path; legacy replay remains unchanged.

Validation reports selected-joint cm² MSE and metre MPJPE. Visualizations convert
positions to metres, include End Sites, and show canonical training trajectories,
not source-heading-restored exports. Checkpoint metadata distinguishes this path
from legacy checkpoints and fingerprints saved statistics and skeleton banks.
Legacy checkpoints and external reference pretrained checkpoints are not directly
loadable as these new method checkpoints.

## Validation and deliberate limits

`tests/test_pan_reference.py` covers strict boundaries, signs, local velocity,
window-local smoothing, saved precision/splits, End Site integration/gradients,
selected losses, separate root/yaw losses, exactly two destination-batch routes,
raw discriminator scores/inputs, zero-speed singularity, and one synthetic CPU
optimizer step. Existing non-root-loss PAN regression tests cover legacy behavior.

Optional differential checks (read-only external inputs):

```sh
PAN_REFERENCE_ROOT=/path/to/pan-motion-retargeting \
PAN_REFERENCE_BVH_ROOT=/path/to/motion-retargeting/data/raw \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
pytest tests/test_pan_reference.py tests/test_adapter_preprocessing.py
```

They check real archive channels/normalization exactly, full End Site FK and its
gradients, two real BVH windows against shipped archives, and the actual reference
`backward_G` loss decomposition and generator gradients on common forward tensors.
The unmodified `backward_G` AST is isolated from unrelated obsolete Mixamo imports
(`numpy.core.umath_tests`); discriminator weights are copied for score/gradient
parity. This is not full learned-checkpoint forward/update parity.

The existing LAFAN1/Dog example configs still select the legacy min-domain
sampler/partial batches rather than human-driven dog recycling/drop-last. An
opt-in `primary_recycle` sampler is now available (used by the separately labeled
CMU/AMASS experiment), but these examples have not silently changed schedules.
Still unmatched/out of scope: historical RNG/initialization and
optimizer-step exposure; full pretrained weight import;
full-clip heading restoration/export/IK cleanup; convergence and visual quality.
The skeleton encoder still stores a masked zero root-offset input column, so its
parameter shape/initialization differs from the reference. The example learning
rate/batch/epoch targets do not make the unchanged sampler a schedule reproduction.
No full training or Isaac Lab run is needed or performed for these checks.

## Generic units, caches, and unsupported equivalence

All generic MotionCollections use **metres** for both offsets and root translation.
AMASS no longer multiplies metre SMPL offsets by 100 while leaving translation in
metres. CMU converts both ASF lengths and AMC translations using 0.0254 metres per
inch (including the ASF length-unit divisor). BANDAI, Mixamo and generic LAFAN1/Dog
already convert cm to metres and keep that contract. Reference PAN tensors alone
use original-frame cm, bypassing generic conversions.

AMASS/CMU now use `amass.metres-v2.pt` / `cmu.metres-v2.pt`; generic LAFAN1/Dog use
`*.reference-split-metres-v2.pt`. Old `.pt` files are preserved and never rescaled,
overwritten, or reused by these adapters. Fresh extraction can require the usual
body-part labeling. Generic PAN checkpoint metadata now fingerprints skeletons
and statistics: older checkpoints without those hashes are rejected rather than
restoring cm/mixed-unit banks under a metre label. Retrain with consistent units;
do not relabel or blindly rescale a historical checkpoint. AMASS extraction options
(model type/beta count) and arbitrary raw-file edits still require deliberate cache
management; these versioned names are not general content-addressed raw caches.

Neither PAN reference pipeline defines an AMASS, ASF/AMC CMU, or BANDAI benchmark
with matching splits/statistics/correspondences. They remain **generic experiments**,
not reference-equivalent training setups. The existing `pan.yaml` still selects
AMASS/CMU; its objectives and hyperparameters have not been re-tuned for these fixes.

Mixamo reference preprocessing is **not** the LAFAN1/Dog contract: it uses parent
rotations per edge plus absolute root XYZ (no yaw channel), [::2] subsampling,
64/32 windows, per-character FP32 sample std before a final 5% holdout, and optional
time reversal. Local Mixamo adapters still supply joint rotations in metre
MotionCollections to generic PAN, not that reference edge/absolute-position model.
A separate Mixamo model path is intentionally not implemented. Only synthetic
Mixamo adapter fixtures were tested because no real Mixamo corpus was available.
No adapter selection alone promises identical training results.

For an explicitly adapted human/human experiment borrowing this objective and
preprocessing, see `docs/pan_cmu_amass_lafan_style.md` and
`configs/methods/pan_cmu_amass_lafan_style.yaml`. It is not a new published reference
benchmark. For side-by-side source/adapter/training-coordinate and exported-NPZ
inspection without Isaac Lab, see `docs/visualize.md`.

Known pre-existing regression issue: `tests/test_pan_root_loss.py` expects an
integrated, height-normalized legacy root loss, but that term is already absent
at the base revision. Its six failures are unrelated and are not repaired by
changing legacy behavior; the new reference root term is a separate normalized
XYZ+yaw MSE, not that older trajectory objective.

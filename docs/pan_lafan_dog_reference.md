# Opt-in LAFAN1/Dog reference semantics

Use `configs/methods/pan_lafan_dog_reference.yaml`, replacing both
`reference_archive.directory` values with the directory containing the **trusted**
reference `humtrain.npz`, `humtest.npz`, `humstats.npz`, `dogtrain.npz`,
`dogtest.npz`, and `dogstats.npz`. These archives contain pickled skeleton dictionaries;
do not use untrusted files. There is no runtime dependency on the reference source tree.

This is a bounded compatibility path, **not a full reproduction or a convergence
claim**. Existing `configs/methods/pan.yaml`, raw adapters/caches, and AMASS/CMU
behavior are unchanged. Do not mix raw sources/params with `reference_archive`.
Exactly one `lafan1` archive domain and one `dog` archive domain are required;
domain group names can be arbitrary. No raw-data directory renaming is necessary.

## Data contract

- Explicit archive train/test sets (6566/2074 human, 2769/779 dog in the shipped
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
lookahead displacement is rotated by the inverse current heading. These helpers
do not regenerate/load raw datasets or repair generic adapter caches.

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
pytest tests/test_pan_reference.py
```

They check real archive channels/normalization exactly, full End Site FK and its
gradients, two real BVH windows against shipped archives, and the actual reference
`backward_G` loss decomposition and generator gradients on common forward tensors.
The unmodified `backward_G` AST is isolated from unrelated obsolete Mixamo imports
(`numpy.core.umath_tests`); discriminator weights are copied for score/gradient
parity. This is not full learned-checkpoint forward/update parity.

Still unmatched/out of scope: the trainer's min-domain sampler/partial batches
versus human-driven dog recycling/drop-last; historical RNG/initialization and
optimizer-step exposure; full pretrained weight import; raw-cache rebuilding;
full-clip heading restoration/export/IK cleanup; convergence and visual quality.
The skeleton encoder still stores a masked zero root-offset input column, so its
parameter shape/initialization differs from the reference. The example learning
rate/batch/epoch targets do not make the unchanged sampler a schedule reproduction.
No full training or Isaac Lab run is needed or performed for these checks.

Known pre-existing regression issue: `tests/test_pan_root_loss.py` expects an
integrated, height-normalized legacy root loss, but that term is already absent
at the base revision. Its six failures are unrelated and are not repaired by
changing legacy behavior; the new reference root term is a separate normalized
XYZ+yaw MSE, not that older trajectory objective.

# PAN joint motion representation

## Layout and kinematics

For J joints, PAN uses `[T, 4J + 4]`: J local `(w,x,y,z)` quaternions in topology order, then
world-space root displacement XYZ and yaw increment in radians **per sampled frame**, not per second.
Root orientation retains world heading. Leaf rotations are retained and supervised by reconstruction,
even though they cannot change joint positions without an outgoing bone. Data conversion keeps the
existing quaternion signs, resampling, window boundaries and statistics policy. A clip of T frames
provides T-1 velocity-bearing poses.

The network has J+1 tokens. The root quaternion and velocity token are accessible to all body parts;
other joint tokens follow their labeled membership. Skeleton masks select the corresponding non-root
joint offsets, never edge indices or an always-on last joint. The existing `[K,J,3]` offset storage is
retained, with its root column masked out of the skeleton encoder.

FK uses `global_rotation[j] = global_rotation[parent] * local_rotation[j]` and
`bone[j] = rotate(global_rotation[parent], offset[j])`. Branches therefore share a single parent's
rotation. Root position at t is the sum of XYZ displacements **before** t, starting at zero for each
window. Yaw is not integrated again: heading is already in the root quaternion. Bone-space discriminator
inputs, root-relative position losses, world-space visualization/end-effector velocities and the
integrated-root loss all use this same FK. Their shapes remain `[B,T,J,3]`.

PAN has no augmentation flag or reversal path. Training, holdout and subsets preserve temporal order.
Random target-bank selection, data-loader shuffling and model dropout are not data augmentation and
remain unchanged. The independent SAN method is not modified.

## Comparison with the actual LAFAN1/dog reference

Checked against `data_preprocess/Lafan1_and_dog/{extract,datasetserial}.py`,
`utils/{utils,rotation,data_utils}.py` and `models/{Intergrated,networks}.py` in the authors' source at
commit `5a51d929fbb3c804494f1d3c58e4558c3be8e35d`.

- **Aligned:** all J joint quaternions plus XYZ velocity and yaw; parent-rotation FK (`quat_fk`, not the
  Mixamo edge FK in `models/Kinematics.py`); direct joint motion masks with globally visible root and
  velocity; non-root joint offset membership; no temporal reversal augmentation.
- **Heading differs deliberately:** the reference removes heading per frame and rotates root velocity
  into that heading-local frame. Human heading comes from the root's local +Y axis projected onto XZ;
  dog heading uses joint 2 minus root, Gaussian-smoothed with width 8. This repository keeps world
  heading, world XYZ displacement and its existing +Z-based Y-up heading extractor. Replacing those
  conventions would be an additional preprocessing/trajectory experiment, not required for joint FK.
- **Root integration differs deliberately:** reference `static2motion` sums through the current frame,
  starting with velocity[0], and its training FK does not apply yaw to reconstruct a world trajectory.
  This implementation preserves the exclusive sum and world heading so poses, contacts, root loss and
  visualization remain in one physical frame. Differential tests subtract the current displacement
  from reference output to compare the same convention.
- **Signs and endpoints differ:** reference dataset loading forces quaternion w >= 0 and repeats the last
  yaw increment within each pre-extracted window. This implementation retains adapter quaternion signs
  and computes forward increments before windowing, dropping the clip's final pose. No new sign or
  boundary policy is introduced.
- **Offsets and topology:** reference skeleton encoders physically omit the root offset; here the full
  J-row cache/buffer remains, with a zero mask on the root column. Existing six-part partition validation,
  adapters, end-effector selection and units are retained. No reference `with_end` topology expansion,
  discriminator redesign, loss replacement, or optimizer/schedule change is included.

Thus this change aligns joint representation/hierarchy, not every aspect of the human/dog experiment.
The retained root conventions are intentional scope choices, not an inherent limitation of joint FK.

## Compatibility and migration

Old PAN motion channels had `4(J-1)+4` values; the new layout has `4J+4`. Normalization statistics,
attention/shortcut/decoder masks and their corresponding weights have different shapes/semantics.
Metadata marks the joint representation, root convention, mask version and disabled augmentation;
loading an old checkpoint fails explicitly rather than silently reinterpreting weights. There is no
weight or optimizer migration: build new PAN windows/statistics and start fresh training.

Existing adapter `MotionCollection` caches already store all joint-local rotations, root positions,
parent-relative offsets and topology, so they remain usable. Any external materialized edge-channel
windows/statistics must be regenerated from those caches or original motion. Inverting edge channels
cannot recover discarded leaf rotations or reconcile independently predicted branch rotations.
Published LAFAN1/dog checkpoints are not directly compatible with this repository's architecture,
preprocessing, parts or units either.

The separate [skeleton-encoder offset-unit conversion](https://github.com/tonyroumi/motion-retargeting/pull/25)
is independent of this change and was still open during validation. No conversion is assumed or
introduced here: persistent offsets/FK stay in metres and the existing encoder call remains unchanged.
If that conversion lands, retain its encoder-only scaling and metadata when integrating.

## Validation

`PAN_REFERENCE_ROOT=/path/to/pan-motion-retargeting pytest tests/test_pan_*.py` exercises joint/leaf
channels, branching FK, root/world velocities, masks, deterministic unaugmented splits, checkpoint
rejection and a small two-domain CPU optimizer/validation smoke. Optional differential tests execute
the actual LAFAN1/dog FK and mask helpers. This is not training convergence, a full-dataset run, GPU
validation or evidence of retargeting quality; those require a separate fresh-training evaluation.

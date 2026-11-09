# motion-retargeting

## Installation

```bash
pip install -e .
```

This installs the `motion_retargeting` package (from `src/`) in editable mode.

## Usage

```bash
python scripts/train.py source=<character> target=<character>
python scripts/retarget.py --model <checkpoint> --source-skeleton <path> --target-skeleton <path> --motion <path>
```

## PAN

There is one PAN implementation, and `configs/methods/pan.yaml` is its only training config. Where the
official code and the paper disagree, it follows the official code: the Mixamo path of
[`hlcdyy/pan-motion-retargeting`](https://github.com/hlcdyy/pan-motion-retargeting) at commit `5a51d92`.
That means edge rotations with heading kept, root velocity (XYZ and yaw rate, as an additional joint; the official code uses the absolute root), full-channel normalization, masked networks,
positional patch discriminators, the official objective, and generator-then-discriminator updates. The
trainer settings are the published Mixamo command: batch size 128, learning rate 1e-3, 1001 epochs, no
schedule.

```bash
python scripts/train.py --config configs/methods/pan.yaml
```

Checkpoints record the official commit, representation, units, normalization scope, mask version, skeleton
banks and endpoints. Loading refuses an older or otherwise incompatible checkpoint. **PAN checkpoints saved
before this implementation do not load.** Checkpoints from before the skeleton-encoder centimetre
conversion are also rejected: their metadata lacks `skeleton_encoder_offset_units: centimeters`.
Weights have the same shapes, but feeding centimetres to weights trained on metres changes predictions;
there is no automatic weight or optimizer migration. Start a new run, or use the old code for old checkpoints.
Do not rescale cached offsets or checkpoint buffers: they remain in metres.

The offset scale was checked against official commit `5a51d92`: `outer_utils/BVH_mod.py` reads raw
BVH offsets, `data_preprocess/Mixamo/bvh_parser.py` selects joint rows, and `combined_motion.py`
stacks them without unit conversion. `models/architecture_mixamo.py` passes them straight to
`SkeletonEncoder`; `models/networks.py` applies the three linear layers and fuses their code as
`z + s / 100`. The Mixamo `Aj.bvh` asset has approximately 32.6-unit upper/lower leg offsets,
consistent with centimetres, not metres (BVH itself has no unit tag). PAN now multiplies its metre
buffers by 100 only at the skeleton-encoder call. The decoder's `/100` is unchanged; it scales a
learned code, not raw offsets, so it is not a second physical-unit conversion.

Focused tests cover this boundary and unchanged physical computations. Whether it improves training
stability has **not** been established; that requires matched training runs, not just forward/backward tests.

### What it does not reproduce

`pan.yaml` runs on the AMASS and CMU sources, so it does not reproduce the paper's dataset-level scores, and
no such claim should be made without an actual matched evaluation. The remaining differences:

- **Sources and units.** AMASS and CMU in metres, not Mixamo's BVH assets. Skeleton-encoder offsets
  are converted to centimetres; FK, heights, losses, discriminator positions and metrics retain their
  existing physical units. This does not reproduce the reference's centimetre-valued physical positions.
- **Characters.** A skeleton bank, meaning clips with exactly equal offsets, stands in for a Mixamo
  character. It does not identify a capture subject, and each bank gets its own statistics.
- **Sampling.** Each domain's motion is sampled independently, not by shared index. Frames are resampled on
  a timestamp grid, not by Mixamo's `::2`.
- **Holdout.** As in the official code, statistics are fit before the 95/5 holdout, so it is a reproduction
  diagnostic, not clean evidence of generalization.
- **Metrics.** There is no paired ground truth. `*_mpjpe_m` is Euclidean error in metres, and
  `*_squared_error_over_height2` is the official metric's form on reconstruction and cycle only. Neither
  is paired retarget accuracy.

### Official quirks kept on purpose

These are official-code behavior, reproduced for parity and not because they are scientifically intended.
Repairing any of them would be a different experiment.

- **Skeleton offset masks.** Edge indices select from a flattened joint offset tensor, and the always-on
  last column reads the last joint, not the root.
- **Contact selection.** Contacts are chosen on generated velocity below `0.008`. An empty selection is NaN
  in the official code; here training stops with an error before either optimizer steps, and no zero-contact
  variant is applied.

One intentional deviation: the temporal conv masks (encoder shortcut and final decoder layer) convert the
manually labeled joint parts to edge indices once. The official code's second conversion, an artifact of mutating
shared lists, is not reproduced.

The differential tests compare the other masks and layers against the official source, when it is available:

```bash
PAN_REFERENCE_ROOT=<official checkout> pytest tests/test_pan_*.py
```

To train on the PAN authors' datasets (Mixamo, LAFAN1 and the dog), see [docs/author_datasets.md](docs/author_datasets.md).

## SAN

There is one SAN implementation, `SANMethod`, and one config, `configs/methods/san.yaml`. It computes what the
official code does at `DeepMotionEditing/deep-motion-editing` commit `74fff8f`, which is the source of truth where
it disagrees with the paper: edge rotations and absolute root, full-channel normalization, the official token
graph, bone-space discriminators, the official objective and generator-then-discriminator updates.

```bash
python scripts/train.py --config configs/methods/san.yaml
```

Checkpoints record the official commit, representation, units, normalization scope, kept accidents, skeleton
banks and endpoints. Loading refuses any other computation or setup. **SAN checkpoints saved before this
computation do not load.**

**An empty contact selection stops training.** Contacts are selected on the generated motion. If a route has
no generated end-effector sample below the threshold, the official code trains on `NaN`; here training stops
with an error before either optimizer steps. Each route logs `<source>_to_<destination>/contact_fraction`, so a
run approaching zero is visible first. Whether that fraction levels off in long runs is not established.

### What this does not reproduce

It runs on the AMASS and CMU sources, so it does not reproduce the paper's dataset-level scores, and no such
claim should be made without an actual matched evaluation. The published errors are not the target of a model
trained from this commit either, because the published weights ignore the skeleton input. The remaining
differences:

- **Sources and units.** AMASS and CMU in metres, not Mixamo's BVH assets. Offsets go to the static path in
  centimetres, so the skeleton term keeps its strength, and to FK and the discriminators in metres.
- **Characters.** A skeleton bank, meaning clips with exactly equal offsets, stands in for a Mixamo character.
  It does not identify a capture subject, and each bank gets its own statistics. Startup prints the windows per
  bank, because a thin bank's statistics describe one or two clips.
- **Sampling.** Each domain's motion is sampled independently, not by shared index, so the official truncation
  of its longer group is not reproduced. Frames are resampled on a timestamp grid, not by `::2`.
- **Training length.** A step budget, not an epoch count: `num_epochs: 1539` is the official 160,008 steps at 104
  steps per epoch. Recompute it if the window count changes.
- **Empty contact selection.** Stopping instead of `NaN`, as above.
- **Holdout.** As in the official code, statistics are fit before the 95/5 holdout, so it is a reproduction
  diagnostic, not clean evidence of generalization.
- **Metrics.** There is no paired ground truth. `reconstruction_mpjpe_m` is Euclidean error in metres, and
  `reconstruction_squared_error_over_height2` is the official metric's form on reconstruction only. Neither is
  paired retarget accuracy.

### Accidents of the official code, kept on purpose

These are official behavior, reproduced for parity and not because they are scientifically intended. Repairing
any of them would be a different experiment, and should be labeled as one.

- **Global token's neighbors.** It reads the neighbors of edge 0, not of the root, and never itself. The
  maintainers have called this a bug (issue 30). It is still kept, because every model trained from this commit
  has it.
- **Joint rows through edge masks.** The static path feeds J joint rows through masks indexed by token, so a token
  reads a joint other than its edge's child, and the global slot reads the last joint. AMASS's joint numbering is
  not depth-first, so on AMASS this crosses chains.
- **Trained discriminator structure.** The discriminator's convolution masks and pooling matrices are trained.
  After one step both pooling matrices are fully dense.
- **Contact selection.** Contacts are chosen on generated velocity below `0.008`, not on the source's.

The differential tests compare against the official source, when it is available:

```bash
SAN_REFERENCE_ROOT=<deep-motion-editing checkout at 74fff8f>/retargeting pytest tests/test_san_*.py
```

## Acknowledgments

This project builds on ideas and methods introduced in
**Skeleton-Aware Networks for Deep Motion Retargeting** by Aberman et al. (2020).


## Citation

```bibtex
@article{aberman2020skeleton,
  author = {Aberman, Kfir and Li, Peizhuo and Lischinski, Dani and Sorkine-Hornung, Olga and Cohen-Or, Daniel and Chen, Baoquan},
  title = {Skeleton-Aware Networks for Deep Motion Retargeting},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {39},
  number = {4},
  pages = {62},
  year = {2020},
  publisher = {ACM}
}
```
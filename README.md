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

## PAN recipes

PAN has two recipes, selected by `method.params.recipe`. Their weights are not interchangeable.

| Recipe | Config | What it is |
| --- | --- | --- |
| `legacy` (default) | `configs/methods/pan.yaml` | The existing paper-shaped adaptation, on heading-free joint rotations and root velocity. |
| `mixamo_reference` | `configs/methods/pan_reference.yaml` | The computation of the official PAN Mixamo path at commit `5a51d92`: edge rotations and absolute root, full-channel normalization, masked networks, positional patch discriminators, the reference objective and generator-then-discriminator updates. |

```bash
python scripts/train.py --config configs/methods/pan_reference.yaml
```

Reference-recipe checkpoints record the recipe, reference commit, representation, units, normalization
scope, mask recipe, skeleton banks and endpoints. Loading refuses a legacy or otherwise incompatible
checkpoint. **Existing PAN checkpoints do not match the new recipe.**

### What `mixamo_reference` does not reproduce

It runs on the AMASS and CMU sources, so it does not reproduce the paper's dataset-level scores, and no
such claim should be made without an actual matched evaluation. The remaining differences:

- **Sources and units.** AMASS and CMU in metres, not Mixamo's BVH assets. Offsets are never rescaled to
  compensate for the reference's `/100` on skeleton codes.
- **Characters.** A skeleton bank, meaning clips with exactly equal offsets, stands in for a Mixamo
  character. It does not identify a capture subject, and each bank gets its own statistics.
- **Sampling.** Each domain's motion is sampled independently, not by shared index. Frames are resampled on
  a timestamp grid, not by Mixamo's `::2`.
- **Holdout.** As in the reference, statistics are fit before the 95/5 holdout, so it is a reproduction
  diagnostic, not clean evidence of generalization.
- **Metrics.** There is no paired ground truth. `*_mpjpe_m` is Euclidean error in metres, and
  `*_squared_error_over_height2` is the reference metric's form on reconstruction and cycle only. Neither
  is paired retarget accuracy.

### Reference quirks kept on purpose

These are reference-HEAD behavior, reproduced for parity and not because they are scientifically intended.
Repairing any of them would be a different experiment.

- **Double-converted temporal masks.** The reference converts its parts to edge indices twice, shifting the
  encoder shortcut and final decoder masks by one index. An edge left uncovered reads all parts. AMASS's
  joint numbering is not depth-first, so on AMASS this shift crosses chains: the left leg's masks read spine
  edges, for example.
- **Skeleton offset masks.** Edge indices select from a flattened joint offset tensor, and the always-on
  last column reads the last joint, not the root.
- **Contact selection.** Contacts are chosen on generated velocity below `0.008`. An empty selection is NaN
  in the reference; here training stops with an error before either optimizer steps, and no zero-contact
  variant is applied.

The differential tests compare against the reference source, when it is available:

```bash
PAN_REFERENCE_ROOT=<reference checkout> pytest tests/test_pan_reference_*.py
```

To train on the PAN authors' datasets (Mixamo, LAFAN1 and the dog), see [docs/author_datasets.md](docs/author_datasets.md).

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
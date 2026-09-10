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

## Acknowledgments

This project builds on ideas and methods introduced in
**Skeleton-Aware Networks for Deep Motion Retargeting** by Aberman et al. (2020).

WIP JEPA APPROACH INCOMING.

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
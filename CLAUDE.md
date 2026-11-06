# Project Overview

`motion_retargeting` is a skeleton-aware deep motion retargeting framework, building on
**Skeleton-Aware Networks for Deep Motion Retargeting** (Aberman et al., 2020). It loads
motion capture data from heterogeneous sources (AMASS/SMPL-H, BANDAI-Namco BVH, ...) through
per-dataset adapters, normalizes everything into a shared `Motion`/`Skeleton` tensor
representation, and builds fixed-length, z-score normalized training windows from it.
A JEPA-based approach is planned as a follow-up direction (see README).

# Architecture

- `motion_retargeting/data/adapters/`
  - `base.py` — `MotionSourceAdapter`, the abstract base for dataset-specific adapters:
    raw/cache directory setup, skeleton/motion disk caching
  - `amass.py` — `AMASSAdapter`, extracts skeleton + motion from the SMPL-H `BodyModel`
    (via `human_body_prior`)
  - `bandai.py` — `BANDAIAdapter`, extracts skeleton + motion from BVH files (via `bvhio`)
- `motion_retargeting/data/builder.py`
  - `MotionDatasetBuilder` / `MotionDatasetBuilderConfig` — turns raw `Motion` clips from
    one or more adapters into fixed-length, z-score normalized training windows
- `motion_retargeting/data/dataset.py`
  - `MotionDataset` (`torch.utils.data.Dataset`) — holds the stacked motion input tensor
    (local joint rotations with global velocity / angular velocity appended as a final
    "joint" row, normalized as one tensor) and skeleton offsets, plus `FeatureStats` for
    denormalizing model output
- `motion_retargeting/data/representations/`
  - `MotionCollection`, `MotionData`, `SkeletonTopology` — plain tensor dataclasses shared
    across adapters, the builder, and kinematics; this is the common interchange format for
    the whole pipeline
- `motion_retargeting/kinematics/`
  - `forward.py` — `ForwardKinematics`, batched forward kinematics for a tree-structured
    skeleton (quaternion convention is `(w, x, y, z)` throughout)
  - `quaternions.py` — pure quaternion algebra primitives (`quat_normalize`, `quat_mul`,
    `quat_apply`, `quat_conjugate`)
- `motion_retargeting/modules/`
  - `base.py` — `MLP` building block (depends on the sibling `moveitmoveit` project for
    activations)
  - `positional_encoding.py` — sinusoidal transformer positional encoding
- `motion_retargeting/config/paths.py`
  - `PATHS` — the single source of truth for the project's data/asset directory layout

# Coding Conventions

- Python 3.10+
- PyTorch
- Avoid unnecessary abstractions.
- Do not modify unrelated code.
- Preserve existing type hints.
- Prefer small focused classes/functions.

# Environment

PyTorch
Ubuntu Linux

Work happens in the `retargeting` conda environment:

conda activate retargeting

Install the package (editable) and its dependencies with:

pip install -e .

Run tests with pytest.

# Working Guidelines

## Scope Control

- Do exactly what was requested and no more.
- Make the smallest change necessary to satisfy the request.
- Do not refactor, rename, reorganize, optimize, or clean up unrelated code.
- Do not change APIs, interfaces, tensor shapes, configuration structure, or behavior unless explicitly requested.
- Do not introduce new abstractions, helper classes, utilities, or dependencies unless they are necessary for the requested change.
- Do not modify nearby code merely because it could be improved.
- Preserve the existing coding style and architecture unless the task specifically asks to change them.

If you notice an unrelated bug, design issue, or possible improvement:
- Mention it separately.
- Do not fix it unless explicitly asked.

If the requested change appears to require a broader modification than expected:
- Explain why.
- Describe the minimum broader change required.
- Do not proceed with the broader change unless it is clearly necessary to complete the request.

## Before Editing

Before making significant changes:

1. Inspect the relevant implementation and its dependencies.
2. Determine the exact scope of the requested change.
3. Identify important tensor shapes, data flow, and behavioral assumptions.
4. Prefer modifying existing code over introducing new architecture.
5. Make the smallest necessary modification.

## Behavioral Preservation

Unless explicitly requested otherwise:

- Preserve existing behavior outside the requested change.
- Preserve function signatures and public APIs.
- Preserve configuration defaults.
- Preserve tensor shapes and device/dtype behavior.
- Preserve existing logging and diagnostics.
- Preserve comments and documentation that remain correct.
- Do not silently change algorithmic semantics.

## Response Discipline

When asked to implement a specific change:

- Focus the response on that change.
- Do not propose multiple alternative architectures unless asked.
- Clearly identify any assumptions you had to make.

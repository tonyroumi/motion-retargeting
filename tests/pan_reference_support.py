"""
Access to the pinned reference PAN source for differential tests. The reference is not a dependency:
set PAN_REFERENCE_ROOT to its checkout (HEAD 5a51d929fbb3c804494f1d3c58e4558c3be8e35d) to run them.

Only the named top-level definitions are executed, avoiding the reference's training scripts, dataset
imports and import side effects.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REFERENCE_ROOT = os.environ.get("PAN_REFERENCE_ROOT")

requires_reference = pytest.mark.skipif(not REFERENCE_ROOT, reason="PAN_REFERENCE_ROOT is not set")

# Parents with the reference's root convention (root parent 0).
FIXTURE_23 = (0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21)
# corps_name_mixamo2_m of the reference's data_preprocess/Mixamo/bvh_parser.py.
FIXTURE_28 = (0, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 14, 18, 19, 20, 21, 14, 23, 24, 25, 26)

MODULES = {
    "utils/utils.py": ["findedgechains", "findbodychain", "getbodyparts", "calselfmask", "build_edge_topology"],
    "models/networks.py": [
        "MotionEncoder",
        "MotionDecoder",
        "SkeletonEncoder",
        "LatentDiscriminator",
        "BodyPartConv",
        "BodyPartMlp",
        "PositionalEncoding",
    ],
    "models/Kinematics.py": ["ForwardKinematics"],
    "loss_function.py": ["Criterion_EE", "GAN_loss", "get_ee"],
}

ARGS = SimpleNamespace(
    transformer_layers=2,
    transformer_latents=32,
    transformer_ffsize=256,
    transformer_heads=2,
    transformer_dropout=0.2,
    transformer_srcdim=4,
    kernel_size=15,
    padding_mode="reflect",
    conv_layers=2,
    upsampling="linear",
    skeleton_info="additive",
    fk_world=0,
    pos_repr="3d",
    rotation="quaternion",
    ee_velo=1,
)


def load_reference() -> SimpleNamespace:
    """ The selected reference definitions, all sharing one namespace. """
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math}
    for path, names in MODULES.items():
        tree = ast.parse((Path(REFERENCE_ROOT) / path).read_text())
        selected = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        exec(compile(ast.Module(selected, []), path, "exec"), namespace)
    return SimpleNamespace(**namespace)


def reference_edges(reference: SimpleNamespace, topology) -> list:
    return reference.build_edge_topology(topology, torch.zeros(len(topology), 3))


def on_cpu():
    """ The reference's encoder pins its masks to CUDA when available, even for a CPU module. """
    return mock.patch.object(torch.cuda, "is_available", return_value=False)


def local_parents(topology) -> torch.Tensor:
    return torch.tensor((-1, *topology[1:]))

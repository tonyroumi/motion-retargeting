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

# Manually labeled body parts of those topologies (GROUP_A_PARTS and GROUP_B_PARTS of the Mixamo adapter):
# left leg, right leg, spine (with the root), head, left arm, right arm.
BODY_JOINTS_23 = {
    0: [1, 2, 3, 4],
    1: [5, 6, 7, 8],
    2: [0, 9, 10, 11, 12],
    3: [13, 14],
    4: [15, 16, 17, 18],
    5: [19, 20, 21, 22],
}
BODY_JOINTS_28 = {
    0: [1, 2, 3, 4, 5],
    1: [6, 7, 8, 9, 10],
    2: [0, 11, 12, 13, 14],
    3: [15, 16, 17],
    4: [18, 19, 20, 21, 22],
    5: [23, 24, 25, 26, 27],
}

# The AMASS and CMU skeletons of configs/methods/pan.yaml, with the body parts one labels them into.
AMASS_BODY_JOINTS = {
    0: [1, 4, 7, 10],
    1: [2, 5, 8, 11],
    2: [0, 3, 6, 9],
    3: [12, 15],
    4: [13, 16, 18, 20],
    5: [14, 17, 19, 21],
}
CMU_BODY_JOINTS = {
    0: [1, 2, 3, 4],
    1: [5, 6, 7, 8],
    2: [0, 9, 10, 11],
    3: [12, 13, 14, 15],
    4: [16, 17, 18, 19],
    5: [20, 21, 22, 23],
}

BODY_JOINTS = {FIXTURE_23: BODY_JOINTS_23, FIXTURE_28: BODY_JOINTS_28}

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


def labeled(topology) -> list:
    """ The topology's labeled body parts as joint lists in key order, a fresh copy on every call. """
    return [list(BODY_JOINTS[topology][key]) for key in sorted(BODY_JOINTS[topology])]


def local_parents(topology) -> torch.Tensor:
    return torch.tensor((-1, *topology[1:]))

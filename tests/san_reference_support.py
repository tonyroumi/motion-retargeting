"""
Access to the pinned official SAN source for differential tests. The reference is not a dependency: set
SAN_REFERENCE_ROOT to the `retargeting/` directory of its checkout
(DeepMotionEditing/deep-motion-editing, HEAD 74fff8fb35e6378351d03fb14ee22fccae28b0bf) to run them.

Only the named top-level definitions are executed, avoiding the reference's training scripts, dataset
imports and import side effects.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REFERENCE_ROOT = os.environ.get("SAN_REFERENCE_ROOT")

# Parents with the official root convention (root parent 0).
FIXTURE_23 = (0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21)
# corps_name_mixamo2_m of the official data_preprocess/Mixamo/bvh_parser.py.
FIXTURE_28 = (0, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 14, 18, 19, 20, 21, 14, 23, 24, 25, 26)

requires_reference = pytest.mark.skipif(not REFERENCE_ROOT, reason="SAN_REFERENCE_ROOT is not set")

MODULES = {
    "option_parser.py": ["get_parser"],
    "models/skeleton.py": [
        "SkeletonConv",
        "SkeletonLinear",
        "SkeletonPool",
        "SkeletonUnpool",
        "build_edge_topology",
        "calc_edge_mat",
        "find_neighbor",
    ],
    "models/enc_and_dec.py": ["Encoder", "Decoder", "AE", "StaticEncoder"],
    "models/vanilla_gan.py": ["Discriminator"],
    "models/Kinematics.py": ["ForwardKinematics"],
    "models/utils.py": ["GAN_loss", "Criterion_EE", "ImagePool", "get_ee"],
    "models/architecture.py": ["GAN_model"],
    "datasets/bvh_parser.py": ["BVH_file"],
}


def local_parents(topology) -> torch.Tensor:
    return torch.tensor((-1, *topology[1:]))


def load_reference() -> SimpleNamespace:
    """ The selected reference definitions, all sharing one namespace, and its default arguments as `args`. """
    namespace = {
        "torch": torch, "nn": nn, "F": F, "np": np, "math": math, "random": random, "argparse": argparse,
        "BaseModel": object,
    }
    for path, names in MODULES.items():
        tree = ast.parse((Path(REFERENCE_ROOT) / path).read_text())
        selected = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        exec(compile(ast.Module(selected, []), path, "exec"), namespace)
    namespace["skeleton"] = SimpleNamespace(find_neighbor=namespace["find_neighbor"])
    namespace["args"] = namespace["get_parser"]().parse_args([])
    return SimpleNamespace(**namespace)


def reference_edges(reference: SimpleNamespace, topology) -> list:
    return reference.build_edge_topology(topology, torch.zeros(len(topology), 3))


def copy_reference_weights(domain, auto_encoder, static_encoder, discriminator) -> None:
    """ Copies official weights into a `SANDomain`, first asserting every mask and pool is equal. """

    def copy_conv(conv, reference_conv) -> None:
        assert torch.equal(conv.mask, reference_conv.mask)
        conv.weight.copy_(reference_conv.weight)
        if conv.bias is not None:
            conv.bias.copy_(reference_conv.bias)
        if reference_conv.add_offset:
            linear, reference_linear = conv.offset_encoder, reference_conv.offset_enc
            assert torch.equal(linear.mask, reference_linear.mask)
            linear.weight.copy_(reference_linear.weight)
            linear.bias.copy_(reference_linear.bias)

    with torch.no_grad():
        encoder_blocks = (domain.motion_encoder.block1, domain.motion_encoder.block2)
        for block, layer in zip(encoder_blocks, auto_encoder.enc.layers):
            copy_conv(block.conv, layer[0])
            assert torch.equal(block.pool.weight, layer[1].weight)

        decoder_blocks = (domain.decoder.block1, domain.decoder.block2)
        for block, layer in zip(decoder_blocks, auto_encoder.dec.layers):
            assert torch.equal(block.unpool.weight, layer[1].weight)
            copy_conv(block.conv, layer[2])

        # The reference's static encoder is linear; a kernel-1 convolution here.
        static_blocks = (domain.offset_encoder.block1, domain.offset_encoder.block2)
        for block, layer in zip(static_blocks, static_encoder.layers):
            assert torch.equal(block.conv.mask[..., 0], layer[0].mask)
            block.conv.weight.copy_(layer[0].weight[..., None])
            block.conv.bias.copy_(layer[0].bias)

        first, second = discriminator.layers
        copy_conv(domain.discriminator.block1.conv, first[0])
        domain.discriminator.block1.norm.load_state_dict(first[1].state_dict())
        assert torch.equal(domain.discriminator.block1.pool.weight, first[2].weight)
        copy_conv(domain.discriminator.block2.conv, second[0])
        assert torch.equal(domain.discriminator.block2.pool.weight, second[1].weight)


def generator_parameter_pairs(domain, auto_encoder, static_encoder) -> list:
    """ (name, our parameter, reference parameter) of a domain's generator networks. """
    pairs = []

    def add_conv(prefix, conv, reference_conv) -> None:
        pairs.append((f"{prefix}.weight", conv.weight, reference_conv.weight))
        pairs.append((f"{prefix}.bias", conv.bias, reference_conv.bias))
        pairs.append((f"{prefix}.offset.weight", conv.offset_encoder.weight, reference_conv.offset_enc.weight))
        pairs.append((f"{prefix}.offset.bias", conv.offset_encoder.bias, reference_conv.offset_enc.bias))

    for index, (block, layer) in enumerate(zip((domain.motion_encoder.block1, domain.motion_encoder.block2), auto_encoder.enc.layers)):
        add_conv(f"encoder{index}", block.conv, layer[0])
    for index, (block, layer) in enumerate(zip((domain.decoder.block1, domain.decoder.block2), auto_encoder.dec.layers)):
        add_conv(f"decoder{index}", block.conv, layer[2])

    # Only the static encoder's first layer is live; the second never gets a gradient on either side.
    block, layer = domain.offset_encoder.block1, static_encoder.layers[0]
    pairs.append(("static0.weight", block.conv.weight, layer[0].weight))
    pairs.append(("static0.bias", block.conv.bias, layer[0].bias))
    return pairs


def discriminator_parameter_pairs(domain, discriminator) -> list:
    """ (name, our parameter, reference parameter) of a discriminator, its trained structure included. """
    first, second = discriminator.layers
    block1, block2 = domain.discriminator.block1, domain.discriminator.block2
    return [
        ("block1.weight", block1.conv.weight, first[0].weight),
        ("block1.mask", block1.conv.mask, first[0].mask),
        ("block1.norm.weight", block1.norm.weight, first[1].weight),
        ("block1.norm.bias", block1.norm.bias, first[1].bias),
        ("block1.pool", block1.pool.weight, first[2].weight),
        ("block2.weight", block2.conv.weight, second[0].weight),
        ("block2.bias", block2.conv.bias, second[0].bias),
        ("block2.mask", block2.conv.mask, second[0].mask),
        ("block2.pool", block2.pool.weight, second[1].weight),
    ]


def assert_close(ours: torch.Tensor, theirs: torch.Tensor, name: str, relative: float = 2e-2) -> None:
    """ Relative to the tensor's own norm: float32 noise on summed large terms, not an absolute scale. """
    ours, theirs = ours.reshape(theirs.shape), theirs
    assert theirs.norm() > 0, f"{name}: the reference tensor is zero, so this comparison says nothing"
    error = ((ours - theirs).norm() / theirs.norm()).item()
    assert error < relative, f"{name}: relative error {error}"

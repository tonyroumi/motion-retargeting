"""
Access to the pinned reference SAN source for differential tests. The reference is not a dependency: set
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
    """ Copies reference weights into a `SANReferenceDomain`, first asserting every mask and pool is equal. """

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

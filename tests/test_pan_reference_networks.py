"""
PAN reference networks: shapes and structure on synthetic topologies, and (with PAN_REFERENCE_ROOT set)
copied-weight output and gradient parity against the pinned reference modules.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

from motion_retargeting.methods.pan.masks import (
    attention_mask,
    humanoid_chains,
    parts_to_parts,
    parts_to_tokens,
    skeleton_offset_mask,
    temporal_conv_mask,
    tokens_to_parts,
)
from motion_retargeting.modules.decoders import ReferenceMotionDecoder
from motion_retargeting.modules.encoders import ReferenceBodyPartEncoder, ReferenceMotionEncoder, ReferenceMotionEncoderConfig
from pan_reference_support import ARGS, FIXTURE_23, FIXTURE_28, load_reference, local_parents, on_cpu, reference_edges, requires_reference

NUM_PARTS = 6
PART_DIM = 32
ATOL, RTOL = 1e-5, 1e-4
FIXTURES = pytest.mark.parametrize("topology", [FIXTURE_23, FIXTURE_28], ids=["23", "28"])


def make_encoder(topology=FIXTURE_23) -> ReferenceMotionEncoder:
    num_tokens = len(topology)
    chains = humanoid_chains(local_parents(topology))
    conv = temporal_conv_mask(chains, num_tokens)
    return ReferenceMotionEncoder(
        num_parts=NUM_PARTS,
        attention_mask=attention_mask(chains, num_tokens),
        shortcut_connectivity=tokens_to_parts(conv, 4, PART_DIM),
        part_connectivity=parts_to_parts(NUM_PARTS, PART_DIM, PART_DIM),
        **asdict(ReferenceMotionEncoderConfig()),
    )


def make_skeleton_encoder(topology=FIXTURE_23) -> ReferenceBodyPartEncoder:
    chains = humanoid_chains(local_parents(topology))
    return ReferenceBodyPartEncoder(
        num_parts=NUM_PARTS,
        input_connectivity=tokens_to_parts(skeleton_offset_mask(chains, len(topology)), 3, PART_DIM),
        part_connectivity=parts_to_parts(NUM_PARTS, PART_DIM, PART_DIM),
    )


def make_decoder(topology=FIXTURE_23) -> ReferenceMotionDecoder:
    conv = temporal_conv_mask(humanoid_chains(local_parents(topology)), len(topology))
    return ReferenceMotionDecoder(
        part_connectivity=parts_to_parts(NUM_PARTS, PART_DIM, PART_DIM),
        output_connectivity=parts_to_tokens(conv, PART_DIM, 4),
    )


def copy_weights(reference_module, module, names: dict) -> None:
    source = reference_module.state_dict()
    result = module.load_state_dict({local: source[remote] for local, remote in names.items()}, strict=False)
    assert not result.unexpected_keys
    assert all(key.endswith("mask") for key in result.missing_keys)


DECODER_LAYERS = {
    "conv1.weight": "layers.0.1.weight",
    "conv1.bias": "layers.0.1.bias",
    "conv2.weight": "layers.1.1.weight",
    "conv2.bias": "layers.1.1.bias",
}
SKELETON_LAYERS = {f"linear{i}.{kind}": f"linear{i}.{kind}" for i in (1, 2, 3) for kind in ("weight", "bias")}


def motion_input(topology=FIXTURE_23, batch: int = 2, frames: int = 64) -> torch.Tensor:
    return torch.randn(batch, frames, 4 * (len(topology) - 1) + 3)


def copy_reference_encoder(reference_encoder, encoder) -> None:
    source = reference_encoder.state_dict()
    target = {
        "part_tokens": source["parameter_part"],
        "embedding.trunk.0.weight": source["joint_pos_encoder.linear1.weight"],
        "embedding.trunk.0.bias": source["joint_pos_encoder.linear1.bias"],
        "embedding.output_head.weight": source["joint_pos_encoder.linear2.weight"],
        "embedding.output_head.bias": source["joint_pos_encoder.linear2.bias"],
    }
    for key, value in source.items():
        if key.startswith("spatialTransEncoder."):
            target[key.replace("spatialTransEncoder.", "transformer.")] = value
    for local, remote in (("conv_shortcut", "conv_residual"), ("conv1", "conv1"), ("conv2", "conv2")):
        target[f"{local}.weight"] = source[f"{remote}.weight"]
        target[f"{local}.bias"] = source[f"{remote}.bias"]

    result = encoder.load_state_dict(target, strict=False)
    assert not result.unexpected_keys
    assert all(key.endswith("mask") for key in result.missing_keys)


def assert_gradients_match(local_module: nn.Module, reference_parameter_grads: dict, names: dict) -> None:
    parameters = dict(local_module.named_parameters())
    for local_name, remote_name in names.items():
        assert torch.allclose(parameters[local_name].grad, reference_parameter_grads[remote_name], atol=ATOL, rtol=RTOL), local_name


class TestEncoder:
    def test_shape(self):
        assert make_encoder()(motion_input()).shape == (2, 16, NUM_PARTS, PART_DIM)

    def test_block_settings(self):
        encoder = make_encoder()
        assert sum(isinstance(module, nn.LayerNorm) for module in encoder.modules()) == 4
        assert len(encoder.transformer.layers) == 2

        layer = encoder.transformer.layers[0]
        assert layer.norm_first is False
        assert layer.self_attn.num_heads == 2
        assert layer.linear1.out_features == 256
        assert layer.activation is nn.functional.gelu
        assert layer.dropout.p == encoder.dropout.p == 0.2

    def test_embedding_and_part_tokens(self):
        encoder = make_encoder()
        assert [encoder.embedding.trunk[0].in_features, encoder.embedding.trunk[0].out_features] == [4, 512]
        assert encoder.embedding.output_head.out_features == 32

        torch.manual_seed(0)
        tokens = torch.cat([make_encoder().part_tokens for _ in range(50)])
        assert tokens.std().item() == pytest.approx(0.1, abs=0.01)

    def test_shortcut_reaches_the_code_only_through_owned_tokens(self):
        encoder = make_encoder().eval()
        encoder.transformer = _Zeros()  # Leave only the shortcut path.
        x = motion_input(batch=1).requires_grad_(True)

        encoder(x)[:, :, 0].sum().backward()
        read_by_left_leg = x.grad.abs().sum(dim=(0, 1))  # Per channel: 4 per edge, then the root's 3.

        # After double conversion the first part owns edges 0, 1, 2 and the root token.
        assert read_by_left_leg[:12].sum() > 0
        assert read_by_left_leg[12:-3].sum() == 0
        assert read_by_left_leg[-3:].sum() > 0

    def test_legacy_encoder_is_untouched(self):
        from motion_retargeting.modules.encoders import MotionEncoder, MotionEncoderConfig

        config = MotionEncoderConfig()
        assert config.embed_dim == 64 and config.embed_hidden_dims == [256, 256]
        assert not any(isinstance(module, nn.LayerNorm) for module in MotionEncoder(
            input_dim=4,
            num_bodies=1,
            joint_body_mask=torch.zeros(4, 4),
            **asdict(config),
        ).modules())


class _Zeros(nn.Module):
    def forward(self, sequence, mask=None):
        return torch.zeros_like(sequence)


@requires_reference
class TestEncoderAgainstReference:
    @FIXTURES
    def test_output_and_gradients_match(self, topology):
        reference = load_reference()
        with on_cpu():
            reference_encoder = reference.MotionEncoder(ARGS, reference_edges(reference, topology), None).eval()
        encoder = make_encoder(topology).eval()
        copy_reference_encoder(reference_encoder, encoder)

        x = motion_input(topology)
        x_local = x.clone().requires_grad_(True)
        x_remote = x.transpose(1, 2).clone().requires_grad_(True)

        local = encoder(x_local)
        remote = reference_encoder(x_remote)
        assert remote.shape == (2, NUM_PARTS * PART_DIM, 16)
        assert torch.allclose(local.flatten(2, 3).transpose(1, 2), remote, atol=ATOL, rtol=RTOL)

        target = torch.randn_like(local)
        (local * target).sum().backward()
        (remote * target.flatten(2, 3).transpose(1, 2)).sum().backward()

        assert torch.allclose(x_local.grad, x_remote.grad.transpose(1, 2), atol=ATOL, rtol=RTOL)
        remote_grads = {name: parameter.grad for name, parameter in reference_encoder.named_parameters() if parameter.grad is not None}
        assert_gradients_match(encoder, remote_grads, {
            "part_tokens": "parameter_part",
            "embedding.trunk.0.weight": "joint_pos_encoder.linear1.weight",
            "embedding.output_head.weight": "joint_pos_encoder.linear2.weight",
            "transformer.layers.1.self_attn.out_proj.weight": "spatialTransEncoder.layers.1.self_attn.out_proj.weight",
            "conv_shortcut.weight": "conv_residual.weight",
            "conv1.weight": "conv1.weight",
            "conv2.bias": "conv2.bias",
        })

    def test_shortcut_alone_matches(self):
        reference = load_reference()
        with on_cpu():
            reference_encoder = reference.MotionEncoder(ARGS, reference_edges(reference, FIXTURE_23), None).eval()
        encoder = make_encoder().eval()
        copy_reference_encoder(reference_encoder, encoder)

        encoder.transformer = _Zeros()
        reference_encoder.spatialTransEncoder = _Zeros()

        x = motion_input()
        local = encoder(x).flatten(2, 3).transpose(1, 2)
        assert torch.allclose(local, reference_encoder(x.transpose(1, 2)), atol=ATOL, rtol=RTOL)

    def test_masks_equal_reference(self):
        reference = load_reference()
        with on_cpu():
            reference_encoder = reference.MotionEncoder(ARGS, reference_edges(reference, FIXTURE_23), None)
        encoder = make_encoder()
        assert torch.equal(encoder.attention_mask, reference_encoder.attention_mask)
        assert torch.equal(encoder.conv_shortcut.mask, reference_encoder.conv_residual.mask[..., :1])
        assert torch.equal(encoder.conv2.mask, reference_encoder.conv2.mask[..., :1])


class TestSkeletonEncoder:
    def test_shape_and_final_activation(self):
        encoder = make_skeleton_encoder()
        code = encoder(torch.randn(3, 23, 3))
        assert code.shape == (3, NUM_PARTS, PART_DIM)

        # LeakyReLU follows all three layers, the last one included.
        activations = []
        encoder.activation.register_forward_hook(lambda module, inputs, output: activations.append(output))
        encoder(torch.randn(3, 23, 3))
        assert len(activations) == 3

    def test_offset_influence_follows_the_mask_with_the_last_offset_always_on(self):
        encoder = make_skeleton_encoder()
        offsets = torch.randn(1, 23, 3).requires_grad_(True)
        code = encoder(offsets)

        mask = skeleton_offset_mask(humanoid_chains(local_parents(FIXTURE_23)), 23)
        for part in range(NUM_PARTS):
            grad = torch.autograd.grad(code[:, part].sum(), offsets, retain_graph=True)[0]
            read = grad[0].abs().sum(dim=-1) > 0
            assert torch.equal(read, mask[part]), part
            assert read[22]


@requires_reference
class TestSkeletonEncoderAgainstReference:
    @FIXTURES
    def test_output_and_gradients_match(self, topology):
        reference = load_reference()
        reference_encoder = reference.SkeletonEncoder(ARGS, reference_edges(reference, topology), None)
        encoder = make_skeleton_encoder(topology)
        copy_weights(reference_encoder, encoder, SKELETON_LAYERS)
        assert torch.equal(encoder.linear1.mask, reference_encoder.linear1.mask)
        assert torch.equal(encoder.linear3.mask, reference_encoder.linear3.mask)

        offsets = torch.randn(2, len(topology), 3)
        local_input = offsets.clone().requires_grad_(True)
        remote_input = offsets.clone().requires_grad_(True)

        local, remote = encoder(local_input), reference_encoder(remote_input)
        assert remote.shape == (2, NUM_PARTS * PART_DIM, 1)
        assert torch.allclose(local.flatten(1), remote[..., 0], atol=ATOL, rtol=RTOL)

        target = torch.randn_like(local)
        (local * target).sum().backward()
        (remote[..., 0] * target.flatten(1)).sum().backward()

        assert torch.allclose(local_input.grad, remote_input.grad, atol=ATOL, rtol=RTOL)
        remote_grads = {name: parameter.grad for name, parameter in reference_encoder.named_parameters() if parameter.grad is not None}
        assert_gradients_match(encoder, remote_grads, SKELETON_LAYERS)


class TestDecoder:
    def test_shape_and_upsampling(self):
        decoder = make_decoder()
        assert decoder(torch.randn(2, 16, NUM_PARTS, PART_DIM)).shape == (2, 64, 4 * 22 + 3)
        assert decoder.ups.mode == "linear" and decoder.ups.align_corners is False

    def test_no_total_width_bottleneck(self):
        decoder = make_decoder()
        assert decoder.conv1.out_channels == NUM_PARTS * PART_DIM

    def test_activation_only_after_the_first_conv(self):
        decoder = make_decoder()
        calls = []
        decoder.activation.register_forward_hook(lambda module, inputs, output: calls.append(1))
        output = decoder(torch.randn(1, 16, NUM_PARTS, PART_DIM))
        assert len(calls) == 1
        # A final unit quaternion normalization or activation would bound or rescale the output.
        assert not torch.allclose(output[..., :4].norm(dim=-1), torch.ones(1, 64))

    def test_forbidden_part_to_output_gradients_are_exactly_zero(self):
        decoder = make_decoder().eval()
        conv = temporal_conv_mask(humanoid_chains(local_parents(FIXTURE_23)), 23)
        code = torch.randn(1, 16, NUM_PARTS, PART_DIM).requires_grad_(True)
        output = decoder(code)

        for edge in range(22):
            grad = torch.autograd.grad(output[:, :, edge * 4:edge * 4 + 4].sum(), code, retain_graph=True)[0]
            read = grad.abs().sum(dim=(0, 1, 3)) > 0
            expected = conv[:, edge] if conv[:, edge].any() else torch.ones(NUM_PARTS, dtype=torch.bool)
            assert torch.equal(read, expected), edge

    def test_multiply_owned_and_unassigned_and_root_outputs(self):
        decoder = make_decoder().eval()
        code = torch.randn(1, 16, NUM_PARTS, PART_DIM).requires_grad_(True)
        output = decoder(code)

        def read_by(channels):
            grad = torch.autograd.grad(output[:, :, channels].sum(), code, retain_graph=True)[0]
            return (grad.abs().sum(dim=(0, 1, 3)) > 0).nonzero().flatten().tolist()

        assert read_by(slice(10 * 4, 11 * 4)) == [2, 3, 4, 5]  # Shared branch edge.
        assert read_by(slice(21 * 4, 22 * 4)) == list(range(NUM_PARTS))  # Unowned after double conversion.
        assert read_by(slice(-3, None)) == list(range(NUM_PARTS))  # Root channels.


@requires_reference
class TestDecoderAgainstReference:
    @FIXTURES
    def test_output_and_gradients_match(self, topology):
        reference = load_reference()
        with on_cpu():
            reference_encoder = reference.MotionEncoder(ARGS, reference_edges(reference, topology), None)
        reference_decoder = reference.MotionDecoder(ARGS, reference_encoder).eval()
        decoder = make_decoder(topology).eval()
        copy_weights(reference_decoder, decoder, DECODER_LAYERS)
        assert torch.equal(decoder.conv2.mask, reference_decoder.layers[1][1].mask[..., :1])

        code = torch.randn(2, 16, NUM_PARTS, PART_DIM)
        local_code = code.clone().requires_grad_(True)
        remote_code = code.flatten(2, 3).transpose(1, 2).clone().requires_grad_(True)

        local, remote = decoder(local_code), reference_decoder(remote_code)
        assert torch.allclose(local, remote.transpose(1, 2), atol=ATOL, rtol=RTOL)

        target = torch.randn_like(local)
        (local * target).sum().backward()
        (remote.transpose(1, 2) * target).sum().backward()

        assert torch.allclose(local_code.grad.flatten(2, 3).transpose(1, 2), remote_code.grad, atol=ATOL, rtol=RTOL)
        remote_grads = {name: parameter.grad for name, parameter in reference_decoder.named_parameters() if parameter.grad is not None}
        assert_gradients_match(decoder, remote_grads, DECODER_LAYERS)

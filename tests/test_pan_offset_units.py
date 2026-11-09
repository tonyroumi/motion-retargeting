"""Centimetres belong only at PAN's skeleton-encoder input, not in its physical buffers."""

import pytest
import torch

from motion_retargeting.methods.pan.config import DomainSpec, PANModelConfig
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.utils import ForwardState, GeneratorOutputs, SourceState


PARENTS = [-1, 0, 0, 0, 3, 3, 3]
BODY_JOINTS = {0: [0, 3], 1: [1], 2: [2], 3: [4], 4: [5], 5: [6]}
OFFSETS = torch.tensor([
    [0.0, 0.0, 0.0], [0.1, -0.5, 0.0], [-0.1, -0.5, 0.0],
    [0.0, 0.5, 0.0], [0.0, 0.3, 0.0], [0.3, 0.2, 0.0], [-0.3, 0.2, 0.0],
])
CHANNELS = 4 * len(PARENTS)


@pytest.fixture
def method():
    torch.manual_seed(7)
    specs = {
        name: DomainSpec(
            parents=torch.tensor(PARENTS),
            offsets=OFFSETS[None] * torch.tensor([scale, scale * 1.5])[:, None, None],
            mean=torch.full((2, CHANNELS), 0.1),
            std=torch.full((2, CHANNELS), 2.0),
            body_joints=BODY_JOINTS,
        )
        for name, scale in (("a", 1.0), ("b", 0.8))
    }
    cfg = PANModelConfig.from_dict({
        "motion_encoder": {
            "embed_dim": 8, "embed_hidden_dim": 16, "num_layers": 1,
            "feedforward_dim": 16, "dropout": 0.0, "kernel_size": 3,
        },
        "motion_decoder": {"kernel_size": 3},
        "motion_discriminator": {"hidden_dim": 8, "kernel_size": 3},
    })
    result = PANMethod(cfg, specs)
    result.eval()
    return result


def batch_for(method):
    batch = {}
    for name, domain in method.model.domains.items():
        physical = domain.offsets.new_zeros(2, 8, CHANNELS)
        physical[..., :-4:4] = 1.0  # Identity edge rotations
        physical[..., -4] = 0.02  # Metres per frame, not centimetres
        bank_ids = torch.tensor([1, 0], device=physical.device)
        batch[name] = {
            "motion": (physical - domain.mean[bank_ids, None]) / domain.std[bank_ids, None],
            "bank_id": bank_ids,
        }
    return batch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_encoder_gets_centimeters_once_and_decoder_keeps_code_division(method, dtype):
    method.model.to(dtype=dtype)
    batch = batch_for(method)
    captured = {name: [] for name in method.model.domains}
    handles = [
        domain.body_part_encoder.linear1.register_forward_pre_hook(
            lambda module, args, name=name: captured[name].append(args[0].detach().clone())
        )
        for name, domain in method.model.domains.items()
    ]
    try:
        first = method.model.forward_generator(batch)
        second = method.model.forward_generator(batch)
    finally:
        for handle in handles:
            handle.remove()

    assert set(first.pairs) == {("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")}
    for name, domain in method.model.domains.items():
        expected_offsets = method.domain_specs[name].offsets.to(dtype=dtype)
        assert len(captured[name]) == 2
        for encoder_input in captured[name]:
            assert encoder_input.dtype == dtype
            assert encoder_input.device == domain.offsets.device
            torch.testing.assert_close(encoder_input, (expected_offsets * 100).flatten(1))
        torch.testing.assert_close(domain.offsets, expected_offsets, rtol=0, atol=0)
        torch.testing.assert_close(first.reconstructions[name], second.reconstructions[name])
        # Includes real encoder biases/nonlinearities: do not scale the learned code instead.
        code = domain.body_part_encoder(expected_offsets * 100)[batch[name]["bank_id"]]
        expected = domain.decoder(first.motion_codes[name] + code[:, None] / 100)
        torch.testing.assert_close(first.reconstructions[name], expected)

    sum(value.square().mean() for value in first.reconstructions.values()).backward()
    for domain in method.model.domains.values():
        grad = domain.body_part_encoder.linear1.weight.grad
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_encoding_does_not_rescale_fk_heights_or_physical_losses(method):
    batch = batch_for(method)
    before = {
        name: domain.bones(batch[name]["motion"], batch[name]["bank_id"])
        for name, domain in method.model.domains.items()
    }
    method.model.forward_generator(batch)
    for name, domain in method.model.domains.items():
        motion, ids = batch[name]["motion"], batch[name]["bank_id"]
        offsets = method.domain_specs[name].offsets[ids]
        bones = domain.bones(motion, ids)
        torch.testing.assert_close(bones, before[name], rtol=0, atol=0)
        expected = offsets[:, None].expand(-1, 8, -1, -1).clone()
        expected[:, :, 0, 0] = torch.arange(8) * 0.02
        torch.testing.assert_close(bones, expected)
        # Foot-to-root plus head-to-root length, still metres.
        heights = offsets[:, [1, 3, 4]].norm(dim=-1).sum(dim=-1)
        torch.testing.assert_close(domain.heights[ids], heights)
        rest = offsets.clone()
        rest[:, 4:] += offsets[:, 3:4]
        world = rest[:, None] + expected[:, :, :1]
        torch.testing.assert_close(domain.fk.world_positions(bones), world)
        velocity = torch.zeros(2, 7, 5, 3)
        velocity[..., 0] = 0.02 * 10 / heights[:, None, None]
        torch.testing.assert_close(method.normalized_end_effector_velocity(name, bones, ids), velocity)

        # Fixed motions isolate physical loss scaling from intentionally changed predictions.
        reconstruction = motion.clone()
        reconstruction[:, 0, -4] += 0.03 / domain.std[ids, -4]
        source = SourceState(
            bank_ids=ids, motion=motion, denormalized=domain.denormalize(motion, ids),
            bones=bones, reconstruction=reconstruction,
            reconstruction_denormalized=domain.denormalize(reconstruction, ids),
            reconstruction_bones=domain.bones(reconstruction, ids),
        )
        state = ForwardState(GeneratorOutputs({}, {}, {}), {name: source}, {})
        losses = method.compute_generator_losses(state)
        error = torch.zeros_like(bones)
        error[:, 1:, 0, 0] = 0.03 / heights[:, None]
        torch.testing.assert_close(losses[f"{name}/root"], method.cfg.root_lambda * error[:, :, 0].square().mean())
        torch.testing.assert_close(losses[f"{name}/position"], method.cfg.position_lambda * error.square().mean())


@pytest.mark.parametrize("old_units", [None, "meters"])
def test_rejects_checkpoints_with_old_encoder_units(method, old_units):
    state = method.state_dict()
    if old_units is None:
        del state["metadata"]["skeleton_encoder_offset_units"]
    else:
        state["metadata"]["skeleton_encoder_offset_units"] = old_units
    with pytest.raises(ValueError, match="skeleton_encoder_offset_units"):
        method.load_state_dict(state)


def test_checkpoint_round_trip_keeps_meter_buffers_without_double_conversion(method, tmp_path):
    batch = batch_for(method)
    before = method.model.forward_generator(batch).reconstructions
    path = tmp_path / "pan.pt"
    torch.save(method.state_dict(), path)
    state = torch.load(path, weights_only=True)
    assert state["metadata"]["physical_units"] == "meters"
    assert state["metadata"]["skeleton_encoder_offset_units"] == "centimeters"
    for _ in range(2):
        method.load_state_dict(state)
        after = method.model.forward_generator(batch).reconstructions
        for name, domain in method.model.domains.items():
            torch.testing.assert_close(domain.offsets, method.domain_specs[name].offsets, rtol=0, atol=0)
            torch.testing.assert_close(after[name], before[name])

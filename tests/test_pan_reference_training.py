"""
PAN reference recipe's routes, objective and update order on synthetic domains, with reference scalar
parity checks that run when PAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from motion_retargeting.methods.pan.config import PANReferenceConfig, PANReferenceDomainSpec, PANReferenceModelConfig
from motion_retargeting.methods.pan.method import PANReferenceMethod

# The AMASS and CMU skeletons of configs/methods/pan_reference.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

BATCH_SIZE = 4
WINDOW = 64
NUM_BANKS = {"group_a": 3, "group_b": 2}
PARENTS = {"group_a": AMASS_PARENTS, "group_b": CMU_PARENTS}


def make_spec(parents, num_banks: int, generator: torch.Generator) -> PANReferenceDomainSpec:
    num_joints = len(parents)
    channels = 4 * (num_joints - 1) + 3

    offsets = 0.1 * torch.randn(num_banks, num_joints, 3, generator=generator)
    offsets[:, 0] = 0.0
    # Rotations stay near identity once denormalized, with banks visibly different from each other.
    mean = torch.zeros(num_banks, channels)
    mean[:, ::4] = 1.0
    mean[:, -3:] = torch.randn(num_banks, 3, generator=generator)
    std = 0.05 + 0.1 * torch.rand(num_banks, channels, generator=generator)

    return PANReferenceDomainSpec(parents=torch.tensor(parents), offsets=offsets, mean=mean, std=std)


def make_specs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {name: make_spec(PARENTS[name], NUM_BANKS[name], generator) for name in PARENTS}


def make_batch(specs, batch_size: int = BATCH_SIZE, window: int = WINDOW, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    batch = {}
    for name, spec in specs.items():
        batch[name] = {
            "motion": torch.randn(batch_size, window, spec.mean.shape[1], generator=generator),
            "bank_id": torch.randint(spec.num_banks, (batch_size,), generator=generator),
        }
    return batch


def make_method(specs=None, seed: int = 0, **cfg) -> PANReferenceMethod:
    torch.manual_seed(seed)
    return PANReferenceMethod(
        model_cfg=PANReferenceModelConfig(),
        domain_specs=specs if specs is not None else make_specs(),
        learning_rate=1e-3,
        device="cpu",
        cfg=PANReferenceConfig(**cfg),
    )


class TestRoutes:
    def test_all_four_pairs_exist(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        assert set(state.pairs) == {
            ("group_a", "group_a"), ("group_a", "group_b"), ("group_b", "group_a"), ("group_b", "group_b"),
        }
        pair = state.outputs.pairs[("group_a", "group_b")]
        assert pair.fake.shape == (BATCH_SIZE, WINDOW, 4 * 23 + 3)
        assert pair.cycle.shape == (BATCH_SIZE, WINDOW, 4 * 21 + 3)

    def test_every_decode_sees_the_fused_code(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        method.eval()

        calls = {"group_a": [], "group_b": []}
        for name, domain in method.model.domains.items():
            domain.decoder.register_forward_pre_hook(lambda module, args, name=name: calls[name].append(args[0]))

        codes = []
        real_decode = method.model.decode

        def spy(domain, motion_code, skeleton_code):
            codes.append((domain, motion_code + skeleton_code[:, None] / 100))
            return real_decode(domain, motion_code, skeleton_code)

        method.model.decode = spy
        method.forward_state(batch)

        # Two reconstructions + four routes of (retarget + cycle).
        assert len(codes) == 10
        seen = {"group_a": 0, "group_b": 0}
        for domain, fused in codes:
            assert torch.equal(calls[domain][seen[domain]], fused)
            seen[domain] += 1

    def test_decoder_input_is_z_plus_s_over_100_on_reconstruction(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        method.eval()

        domain = method.model.domains["group_a"]
        inputs = []
        domain.decoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))

        outputs = method.model.forward_generator(batch)

        codes = domain.body_part_encoder(domain.offsets)[batch["group_a"]["bank_id"]]
        expected = outputs.motion_codes["group_a"] + codes[:, None] / 100
        assert torch.allclose(inputs[0], expected)

    def test_target_draws_cover_banks_uniformly_regardless_of_clips(self):
        method = make_method()
        batch = make_batch(method.domain_specs, batch_size=1)
        counts = torch.zeros(NUM_BANKS["group_a"])
        with torch.no_grad():
            torch.manual_seed(3)
            for _ in range(60):
                outputs = method.model.forward_generator(batch)
                counts[outputs.pairs[("group_b", "group_a")].target_ids] += 1
        assert (counts > 5).all()

    def test_target_statistics_offsets_and_height_are_coupled(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        domain = method.model.domains["group_b"]
        pair = state.outputs.pairs[("group_a", "group_b")]

        denormalized = pair.fake * domain.std[pair.target_ids][:, None] + domain.mean[pair.target_ids][:, None]
        expected = domain.fk.bone_vectors_with_absolute_root(denormalized, domain.offsets[pair.target_ids])
        assert torch.equal(state.pairs[("group_a", "group_b")].fake_bones, expected)
        assert pair.target_ids.equal(state.pairs[("group_a", "group_b")].target_ids)

    def test_same_domain_routes_use_the_drawn_target_too(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs, batch_size=32))
        same = state.pairs[("group_a", "group_a")]
        assert not torch.equal(same.target_ids, state.sources["group_a"].bank_ids)

    def test_target_morphology_never_changes_source_cycle_offsets(self):
        method = make_method()
        batch = make_batch(method.domain_specs)
        method.eval()

        state = method.forward_state(batch)
        source = method.model.domains["group_a"]
        pair = state.outputs.pairs[("group_a", "group_b")]
        bank_ids = batch["group_a"]["bank_id"]

        denormalized = pair.cycle * source.std[bank_ids][:, None] + source.mean[bank_ids][:, None]
        expected = source.fk.bone_vectors_with_absolute_root(denormalized, source.offsets[bank_ids])
        assert torch.equal(state.pairs[("group_a", "group_b")].cycle_bones, expected)

    def test_retarget_and_latent_are_not_detached(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        pair = state.outputs.pairs[("group_a", "group_b")]
        assert pair.fake.requires_grad and pair.fake_code.requires_grad and pair.cycle.requires_grad
        assert state.outputs.motion_codes["group_a"].requires_grad

    def test_real_motion_positions_are_detached(self):
        method = make_method()
        state = method.forward_state(make_batch(method.domain_specs))
        assert not state.sources["group_a"].bones.requires_grad

    def test_visualization_uses_absolute_root_world_positions(self, tmp_path):
        method = make_method()
        method.eval()
        method.visualize(make_batch(method.domain_specs, batch_size=1), tmp_path, fps=30.0)
        assert (tmp_path / "group_a_to_group_b.gif").exists()
        assert (tmp_path / "group_b_to_group_a.gif").exists()

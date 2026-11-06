"""
SAN reference recipe's networks on synthetic banks, with copied-weight reference parity when
SAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from motion_retargeting.methods.san.config import DomainSpec, SANModelConfig, SANReferenceDomainSpec
from motion_retargeting.methods.san.model import SANDomain, SANReferenceDomain
from motion_retargeting.methods.san.utils import select_banks
from pan_reference_support import FIXTURE_23, FIXTURE_28, local_parents
from san_reference_support import copy_reference_weights, load_reference, reference_edges, requires_reference

ROOT = Path(__file__).resolve().parents[1]

AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

WINDOW = 64


def make_spec(parents, num_banks: int = 3, seed: int = 0) -> SANReferenceDomainSpec:
    generator = torch.Generator().manual_seed(seed)
    num_joints = len(parents)
    channels = 4 * (num_joints - 1) + 3
    offsets = 0.1 * torch.randn(num_banks, num_joints, 3, generator=generator)
    offsets[:, 0] = 0.0
    mean = torch.zeros(num_banks, channels)
    mean[:, ::4] = 1.0
    std = 0.01 + 0.1 * torch.rand(num_banks, channels, generator=generator)
    return SANReferenceDomainSpec(parents=torch.tensor(parents), offsets=offsets, mean=mean, std=std)


def make_domain(parents, cfg: SANModelConfig | None = None, seed: int = 0) -> SANReferenceDomain:
    torch.manual_seed(seed)
    return SANReferenceDomain(cfg if cfg is not None else SANModelConfig(), make_spec(parents))


class TestShapes:
    @pytest.mark.parametrize("parents, channels", [(AMASS_PARENTS, 87), (CMU_PARENTS, 95)])
    def test_latent_and_output(self, parents, channels):
        domain = make_domain(parents)
        ids = torch.tensor([0, 2])
        static = select_banks(domain.encode_offsets(), ids)

        latent = domain.encode(torch.randn(2, WINDOW, channels), static)
        assert latent.shape == (2, 112, WINDOW // 4)
        assert domain.decode(latent, static).shape == (2, WINDOW, channels)

    def test_static_levels_cover_every_bank(self):
        domain = make_domain(CMU_PARENTS)
        levels = domain.encode_offsets()
        assert [level.shape[0] for level in levels] == [3, 3, 3]
        assert levels[0].shape == (3, 3 * len(CMU_PARENTS), 1)

    def test_offsets_are_fed_in_centimetres(self):
        domain = make_domain(CMU_PARENTS)
        assert torch.equal(domain.encode_offsets()[0][..., 0], (domain.offsets * 100).flatten(1))

    @pytest.mark.parametrize("parents", [AMASS_PARENTS, CMU_PARENTS])
    def test_discriminator_scores_a_window_of_bones_with_no_global_row(self, parents):
        domain = make_domain(parents)
        scores = domain.discriminate(torch.randn(2, WINDOW, len(parents), 3))
        assert scores.shape == (2, 84, 9)
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_bones_use_each_samples_bank(self):
        domain = make_domain(CMU_PARENTS)
        motion = torch.randn(1, 4, 95)
        one = domain.bones(motion, torch.tensor([0]))
        other = domain.bones(motion, torch.tensor([1]))

        assert one.shape == (1, 4, 24, 3)
        assert not torch.equal(one, other)
        assert torch.allclose(one[..., 0, :], domain.denormalize(motion, torch.tensor([0]))[..., -3:])


class TestKeptJointRowAccident:
    @pytest.mark.parametrize("parents", [AMASS_PARENTS, CMU_PARENTS])
    @pytest.mark.parametrize("joint", [1, 5, 13])
    def test_perturbing_a_joint_changes_exactly_the_tokens_reading_its_index(self, parents, joint):
        domain = make_domain(parents)
        conv = domain.offset_encoder.block1.conv
        channels = conv.out_channels_per_joint

        captured = []
        conv.register_forward_hook(lambda module, args, output: captured.append(output.detach()))

        domain.encode_offsets()
        domain.offsets[:, joint] += 1.0
        domain.encode_offsets()

        changed = (captured[0] != captured[1]).any(dim=(0, 2))
        by_token = changed.reshape(-1, channels).any(dim=1)
        reading = [joint in neighbors for neighbors in conv.adj]
        assert by_token.tolist() == reading


def token_biases(conv):
    """ (token's bias slice, its own bound) per token of a convolution. """
    channels = conv.out_channels_per_joint
    kernel_size = conv.weight.shape[-1]
    for token, neighbors in enumerate(conv.expanded_adj_list):
        yield conv.bias[token * channels:(token + 1) * channels], 1 / len(neighbors) ** 0.5 / kernel_size ** 0.5


class TestBiasInitialization:
    def test_first_encoder_layer_draws_each_token_inside_its_own_bound(self):
        conv = make_domain(AMASS_PARENTS).motion_encoder.block1.conv
        whole_weight_bound = 1 / (conv.in_channels * conv.weight.shape[-1]) ** 0.5

        for bias, bound in token_biases(conv):
            assert bias.abs().max() <= bound

        assert whole_weight_bound == pytest.approx(0.0275, abs=1e-4)
        assert 1 / (32 * 15) ** 0.5 == pytest.approx(0.0456, abs=1e-4)
        assert conv.bias.abs().max() > whole_weight_bound

    @pytest.mark.parametrize("network", ["motion_encoder", "decoder"])
    def test_every_generator_convolution_uses_per_token_bounds(self, network):
        module = getattr(make_domain(CMU_PARENTS), network)
        for block in (module.block1, module.block2):
            for bias, bound in token_biases(block.conv):
                assert bias.abs().max() <= bound

    def test_discriminator_draws_per_token_and_keeps_its_bias_free_first_layer(self):
        discriminator = make_domain(CMU_PARENTS).discriminator
        assert discriminator.block1.conv.bias is None
        for bias, bound in token_biases(discriminator.block2.conv):
            assert bias.abs().max() <= bound

    def test_static_path_keeps_the_whole_weight_bound_the_references_linear_layers_use(self):
        conv = make_domain(CMU_PARENTS).offset_encoder.block1.conv
        assert conv.bias.abs().max() <= 1 / (conv.in_channels * conv.weight.shape[-1]) ** 0.5

    def test_legacy_domain_keeps_the_whole_weight_bound(self):
        torch.manual_seed(0)
        spec = DomainSpec(parents=torch.tensor(AMASS_PARENTS), num_joints=len(AMASS_PARENTS))
        conv = SANDomain(SANModelConfig(), spec).motion_encoder.block1.conv
        assert conv.bias.abs().max() <= 1 / (conv.in_channels * conv.weight.shape[-1]) ** 0.5

    def test_loaded_checkpoints_keep_their_biases(self):
        domain = make_domain(CMU_PARENTS)
        restored = make_domain(CMU_PARENTS, seed=1)
        assert not torch.equal(domain.motion_encoder.block1.conv.bias, restored.motion_encoder.block1.conv.bias)

        restored.load_state_dict(domain.state_dict())
        assert torch.equal(domain.motion_encoder.block1.conv.bias, restored.motion_encoder.block1.conv.bias)


def structure_config() -> SANModelConfig:
    cfg = SANModelConfig()
    cfg.motion_discriminator.train_structure = True
    return cfg


def discriminator_step(domain: SANReferenceDomain) -> None:
    optimizer = torch.optim.Adam(domain.discriminator.parameters(), lr=2e-4)
    bones = torch.randn(4, WINDOW, len(domain.fk.parents), 3)
    optimizer.zero_grad()
    domain.discriminate(bones).mean().backward()
    optimizer.step()


class TestTrainableDiscriminatorStructure:
    def test_flag_comes_from_the_reference_yaml_only(self):
        for name, expected in (("san_reference.yaml", True), ("san.yaml", False)):
            cfg = yaml.safe_load((ROOT / "configs" / "methods" / name).read_text())
            assert SANModelConfig.from_dict(cfg["method"]["model"]).motion_discriminator.train_structure is expected

    def test_parameter_counts(self):
        assert len(list(make_domain(CMU_PARENTS).discriminator.parameters())) == 5
        assert len(list(make_domain(CMU_PARENTS, structure_config()).discriminator.parameters())) == 9

    def test_structure_values_do_not_change_when_registered(self):
        fixed = make_domain(CMU_PARENTS).discriminator
        trained = make_domain(CMU_PARENTS, structure_config()).discriminator
        for fixed_block, trained_block in ((fixed.block1, trained.block1), (fixed.block2, trained.block2)):
            assert torch.equal(fixed_block.conv.mask, trained_block.conv.mask)
            assert torch.equal(fixed_block.pool.weight, trained_block.pool.weight)

    def test_one_step_fills_the_pools_keeps_mask_support_and_leaves_the_generator_alone(self):
        domain = make_domain(CMU_PARENTS, structure_config())
        discriminator = domain.discriminator
        generator_structure = {
            name: value.clone()
            for name, value in domain.named_buffers()
            if not name.startswith("discriminator.")
        }
        assert any(name.startswith("decoder.") and name.endswith(".mask") for name in generator_structure)
        supports = {block: block.conv.mask.detach().clone() != 0 for block in (discriminator.block1, discriminator.block2)}
        masks = {block: block.conv.mask.detach().clone() for block in supports}

        discriminator_step(domain)

        for block, support in supports.items():
            assert (block.pool.weight != 0).all()
            assert torch.equal(block.conv.mask != 0, support)
            assert not torch.equal(block.conv.mask, masks[block])

        for name, value in generator_structure.items():
            assert torch.equal(domain.get_buffer(name), value)

    def test_trained_structure_survives_a_checkpoint_round_trip(self):
        domain = make_domain(CMU_PARENTS, structure_config())
        discriminator_step(domain)

        restored = make_domain(CMU_PARENTS, structure_config(), seed=1)
        restored.load_state_dict(domain.state_dict())

        for block, restored_block in ((domain.discriminator.block1, restored.discriminator.block1), (domain.discriminator.block2, restored.discriminator.block2)):
            assert torch.equal(block.conv.mask, restored_block.conv.mask)
            assert torch.equal(block.pool.weight, restored_block.pool.weight)

    def test_without_the_flag_the_structure_is_not_in_the_state_dict(self):
        state = make_domain(CMU_PARENTS).discriminator.state_dict()
        assert "block1.pool.weight" not in state and "block1.conv.mask" not in state


@requires_reference
class TestAgainstReference:
    @pytest.mark.parametrize("fixture", [FIXTURE_23, FIXTURE_28], ids=["23_joints", "28_joints"])
    def test_copied_weights_match_the_reference(self, fixture):
        reference = load_reference()
        args = reference.args
        edges = reference_edges(reference, fixture)

        torch.manual_seed(0)
        auto_encoder = reference.AE(args, edges)
        static_encoder = reference.StaticEncoder(args, edges)
        discriminator = reference.Discriminator(args, edges)

        domain = make_domain(local_parents(fixture).tolist())
        copy_reference_weights(domain, auto_encoder, static_encoder, discriminator)

        centimetres = domain.offsets * 100
        ours = domain.encode_offsets()
        theirs = static_encoder(centimetres)

        assert torch.equal(ours[0][..., 0], centimetres.flatten(1))
        for our_level, their_level in zip(ours[1:], theirs[1:]):
            assert torch.allclose(our_level[..., 0], their_level, atol=1e-5)

        ids = torch.tensor([0, 1, 2, 0])
        motion = torch.randn(4, WINDOW, 4 * (len(fixture) - 1) + 3)
        static = select_banks(ours, ids)

        latent = domain.encode(motion, static)
        their_static = [level[ids] for level in theirs]
        their_latent, their_decoded = auto_encoder(motion.transpose(1, 2), their_static)
        assert torch.allclose(latent, their_latent, atol=1e-5)
        assert torch.allclose(domain.decode(latent, static), their_decoded.transpose(1, 2), atol=1e-5)

        bones = torch.randn(4, WINDOW, len(fixture), 3)
        assert torch.allclose(domain.discriminate(bones), discriminator(bones), atol=1e-5)

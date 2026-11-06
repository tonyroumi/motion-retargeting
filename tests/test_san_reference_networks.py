"""
SAN reference recipe's networks on synthetic banks, with copied-weight reference parity when
SAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

import pytest
import torch

from motion_retargeting.methods.san.config import SANModelConfig, SANReferenceDomainSpec
from motion_retargeting.methods.san.model import SANReferenceDomain
from motion_retargeting.methods.san.utils import select_banks
from pan_reference_support import FIXTURE_23, FIXTURE_28, local_parents
from san_reference_support import copy_reference_weights, load_reference, reference_edges, requires_reference

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

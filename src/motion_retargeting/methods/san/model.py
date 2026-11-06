"""
SAN networks: skeleton-aware encoders, decoder and discriminator per motion domain.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterator, List

import torch
from torch import nn

from motion_retargeting.methods.pan.kinematics import EdgeForwardKinematics
from motion_retargeting.modules.decoders import SkeletalDecoder
from motion_retargeting.modules.discriminators import SkeletalDiscriminator
from motion_retargeting.modules.encoders import SkeletalEncoder
from motion_retargeting.modules.skeletal import construct_adj

from .config import DomainSpec, SANModelConfig, SANReferenceDomainSpec
from .utils import GeneratorOutputs

# Offsets come in meters, while SkeletalConv's static branch (scaled by 1/100) expects centimeters.
METERS_TO_CENTIMETERS = 100.0


class SANDomain(nn.Module):
    """
    Encoders / decoder / discriminator for a single skeleton domain. Takes motion in the
    decoder layout [B, W, J+1, 4]; the networks see it as [B, (J+1) * 4, W].
    """

    def __init__(self, cfg: SANModelConfig, spec: DomainSpec):
        super().__init__()

        edge_list = spec.edge_list
        adj_list = construct_adj(edge_list)

        self.offset_encoder = SkeletalEncoder(
            adj_list,
            edge_list,
            return_all=True,
            **asdict(cfg.offset_encoder),
        )
        self.motion_encoder = SkeletalEncoder(
            adj_list,
            edge_list,
            **asdict(cfg.motion_encoder),
        )
        self.decoder = SkeletalDecoder(
            self.motion_encoder.pooling_hierarchy,
            **asdict(cfg.motion_decoder),
        )
        self.discriminator = SkeletalDiscriminator(
            self.offset_encoder.pooling_hierarchy,
            **asdict(cfg.motion_discriminator),
        )

    def encode_offsets(self, offsets: torch.Tensor) -> List[torch.Tensor]:
        """ offsets: [B, J, 3] -> static features per encoder level, zero on the global row. """
        global_row = offsets.new_zeros(offsets.shape[0], 1, offsets.shape[-1])
        x = torch.cat([offsets, global_row], dim=1) * METERS_TO_CENTIMETERS
        return self.offset_encoder(x.flatten(1).unsqueeze(-1))

    def encode(self, motion: torch.Tensor, offset_features: List[torch.Tensor]) -> torch.Tensor:
        """ motion: [B, W, J+1, 4] -> latent on the primal skeleton, [B, regions * channels, W / 4]. """
        return self.motion_encoder(motion.flatten(2).transpose(1, 2), offset_features)

    def decode(self, latent: torch.Tensor, offset_features: List[torch.Tensor]) -> torch.Tensor:
        """ latent: [B, regions * channels, W / 4] -> motion [B, W, J+1, 4]. """
        motion = self.decoder(latent, offset_features).transpose(1, 2)
        return motion.reshape(*motion.shape[:2], -1, 4)

    def discriminate(self, positions: torch.Tensor) -> torch.Tensor:
        """ positions: [B, W, J, 3] -> realism scores, on a zero global row (positions carry no global features). """
        global_row = positions.new_zeros(*positions.shape[:2], 1, positions.shape[-1])
        return self.discriminator(torch.cat([positions, global_row], dim=2).flatten(2))


class SANModel(nn.Module):
    """
    One `SANDomain` per motion domain, keyed by the dataset group name
    (e.g. "group_a", "group_b"). Domains share a latent space, so their skeletons must
    reduce to the same primal skeleton and have the same number of end-effectors.
    """

    def __init__(self, cfg: SANModelConfig, domain_specs: Dict[str, DomainSpec]):
        super().__init__()

        self.cfg = cfg
        self.specs = domain_specs
        self.domains = nn.ModuleDict(
            {name: SANDomain(cfg, spec) for name, spec in domain_specs.items()}
        )

        primal_regions = {
            name: len(domain.motion_encoder.pooling_hierarchy[-1].pooled_edges)
            for name, domain in self.domains.items()
        }
        if len(set(primal_regions.values())) > 1:
            raise ValueError(f"Domains must share a primal skeleton, got regions per domain: {primal_regions}")

        end_effectors = {name: len(spec.end_effectors) for name, spec in domain_specs.items()}
        if len(set(end_effectors.values())) > 1:
            raise ValueError(f"Domains must have the same end-effector count, got: {end_effectors}")

    def forward_generator(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> GeneratorOutputs:
        """ batch: {domain: {"motion": [B, W, J+1, 4], "offsets": [B, J, 3]}} """
        domain_a = self.domains["group_a"]
        domain_b = self.domains["group_b"]

        motion_a, offsets_a = batch["group_a"]["motion"], batch["group_a"]["offsets"]
        motion_b, offsets_b = batch["group_b"]["motion"], batch["group_b"]["offsets"]

        # Encode
        skel_a = domain_a.encode_offsets(offsets_a)
        skel_b = domain_b.encode_offsets(offsets_b)

        z_motion_a = domain_a.encode(motion_a, skel_a)
        z_motion_b = domain_b.encode(motion_b, skel_b)

        # Self reconstruction
        group_aa = domain_a.decode(z_motion_a, skel_a)
        group_bb = domain_b.decode(z_motion_b, skel_b)

        # Cross-domain retargeting
        group_ab = domain_b.decode(z_motion_a, skel_b)
        group_ba = domain_a.decode(z_motion_b, skel_a)

        # Re-encode the decoded motions
        z_motion_aa = domain_a.encode(group_aa, skel_a)
        z_motion_ab = domain_b.encode(group_ab, skel_b)
        z_motion_ba = domain_a.encode(group_ba, skel_a)
        z_motion_bb = domain_b.encode(group_bb, skel_b)

        return GeneratorOutputs(
            group_aa=group_aa,
            group_bb=group_bb,
            group_ab=group_ab,
            group_ba=group_ba,
            motion_code_a=z_motion_a,
            motion_code_b=z_motion_b,
            motion_code_aa=z_motion_aa,
            motion_code_ab=z_motion_ab,
            motion_code_ba=z_motion_ba,
            motion_code_bb=z_motion_bb,
        )

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.motion_encoder.parameters()
            yield from domain.offset_encoder.parameters()
            yield from domain.decoder.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.discriminator.parameters()


class SANReferenceDomain(nn.Module):
    """
    Reference-recipe networks of one domain, with its skeleton banks as persistent buffers. Takes motion in
    the data view's layout [B, T, 4E + 3]; the networks see it as [B, 4 * J, T], with the root position
    padded to a full token by one zero channel.
    """

    def __init__(self, cfg: SANModelConfig, spec: SANReferenceDomainSpec):
        super().__init__()

        # A target's identity selects these four together, for same-domain retargets as well.
        self.register_buffer("offsets", spec.offsets)
        self.register_buffer("mean", spec.mean)
        self.register_buffer("std", spec.std)
        self.register_buffer("heights", spec.heights)

        self.fk = EdgeForwardKinematics(spec.parents.tolist())
        self.ee_ids = spec.ee_ids

        edge_list = spec.edge_list
        adj_list = construct_adj(edge_list)

        self.offset_encoder = SkeletalEncoder(
            adj_list,
            edge_list,
            return_all=True,
            **asdict(cfg.offset_encoder),
        )
        self.motion_encoder = SkeletalEncoder(
            adj_list,
            edge_list,
            **asdict(cfg.motion_encoder),
        )
        self.decoder = SkeletalDecoder(
            self.motion_encoder.pooling_hierarchy,
            **asdict(cfg.motion_decoder),
        )
        self.discriminator = SkeletalDiscriminator(
            self.offset_encoder.pooling_hierarchy,
            **asdict(cfg.motion_discriminator),
        )

    def encode_offsets(self) -> List[torch.Tensor]:
        """
        Static features of all K banks, per encoder level. The J joint rows go through masks built for the
        J edge-and-root tokens, with no global row added: token e reads joint e rather than its own edge's
        child, and the last token reads the last joint. A reference accident, kept on purpose.
        """
        return self.offset_encoder((self.offsets * METERS_TO_CENTIMETERS).flatten(1).unsqueeze(-1))

    def encode(self, motion: torch.Tensor, static: List[torch.Tensor]) -> torch.Tensor:
        """ motion: [B, T, 4E + 3], static: per-sample levels -> latent [B, tokens * channels, T / 4]. """
        padded = torch.cat([motion, torch.zeros_like(motion[..., :1])], dim=-1)
        return self.motion_encoder(padded.transpose(1, 2), static)

    def decode(self, latent: torch.Tensor, static: List[torch.Tensor]) -> torch.Tensor:
        """ latent: [B, tokens * channels, T / 4], static: per-sample levels -> motion [B, T, 4E + 3]. """
        return self.decoder(latent, static).transpose(1, 2)[..., :-1]

    def denormalize(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ [B, T, 4E + 3], with each sample's bank statistics. """
        return motion * self.std[bank_ids][:, None] + self.mean[bank_ids][:, None]

    def bones(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Normalized motion -> bone_vectors_with_absolute_root on each sample's bank. """
        return self.fk.bone_vectors_with_absolute_root(self.denormalize(motion, bank_ids), self.offsets[bank_ids])

    def discriminate(self, bones: torch.Tensor) -> torch.Tensor:
        """ bones: [B, T, J, 3], in metres with no global row -> realism scores. BatchNorm removes the unit. """
        return self.discriminator(bones.flatten(2))

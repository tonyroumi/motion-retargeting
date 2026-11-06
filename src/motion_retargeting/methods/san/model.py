"""
SAN networks: skeleton-aware encoders, decoder and discriminator per motion domain.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterator, List

import torch
from torch import nn

from motion_retargeting.modules.decoders import SkeletalDecoder
from motion_retargeting.modules.discriminators import SkeletalDiscriminator
from motion_retargeting.modules.encoders import SkeletalEncoder
from motion_retargeting.modules.skeletal import SkeletalConv, construct_adj

from .config import DomainSpec, SANModelConfig
from .kinematics import EdgeForwardKinematics
from .utils import GeneratorOutputs, PairOutputs, select_banks

# Offsets come in meters, while SkeletalConv's static branch (scaled by 1/100) expects centimeters.
METERS_TO_CENTIMETERS = 100.0


class SANDomain(nn.Module):
    """
    Encoders / decoder / discriminator for a single skeleton domain, with its skeleton banks as
    persistent buffers. Takes motion in the data view's layout [B, T, 4E + 3]; the networks see it as
    [B, 4 * J, T], with the root position padded to a full token by one zero channel.
    """

    def __init__(self, cfg: SANModelConfig, spec: DomainSpec):
        super().__init__()

        # A target's identity selects these four together, for same-domain retargets as well.
        self.register_buffer("offsets", spec.offsets)
        self.register_buffer("mean", spec.mean)
        self.register_buffer("std", spec.std)
        self.register_buffer("heights", spec.heights)

        self.fk = EdgeForwardKinematics(spec.parents.tolist())
        self.end_effectors = spec.end_effectors

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

        # The official convolutions draw biases per token. Its static encoder is linear, with the
        # whole-weight fan-in that the offset encoder's kernel-1 convolutions already use.
        for network in (self.motion_encoder, self.decoder, self.discriminator):
            for module in network.modules():
                if isinstance(module, SkeletalConv):
                    module.init_bias_per_token()

    def encode_offsets(self) -> List[torch.Tensor]:
        """
        Static features of all K banks, per encoder level. The J joint rows go through masks built for the
        J edge-and-root tokens, with no global row added: token e reads joint e rather than its own edge's
        child, and the last token reads the last joint. An accident of the official code, kept on purpose.
        """
        return self.offset_encoder((self.offsets * METERS_TO_CENTIMETERS).flatten(1).unsqueeze(-1))

    def encode(self, motion: torch.Tensor, offset_features: List[torch.Tensor]) -> torch.Tensor:
        """ motion: [B, T, 4E + 3] -> latent on the primal skeleton, [B, regions * channels, T / 4]. """
        padded = torch.cat([motion, torch.zeros_like(motion[..., :1])], dim=-1)
        return self.motion_encoder(padded.transpose(1, 2), offset_features)

    def decode(self, latent: torch.Tensor, offset_features: List[torch.Tensor]) -> torch.Tensor:
        """ latent: [B, regions * channels, T / 4] -> motion [B, T, 4E + 3]. """
        return self.decoder(latent, offset_features).transpose(1, 2)[..., :-1]

    def denormalize(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ [B, T, 4E + 3], with each sample's bank statistics. """
        return motion * self.std[bank_ids][:, None] + self.mean[bank_ids][:, None]

    def bones(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Normalized motion -> bone_vectors_with_absolute_root on each sample's bank. """
        return self.fk.bone_vectors_with_absolute_root(self.denormalize(motion, bank_ids), self.offsets[bank_ids])

    def discriminate(self, bones: torch.Tensor) -> torch.Tensor:
        """ bones: [B, T, J, 3], in metres with no global row -> realism scores. BatchNorm removes the unit. """
        return self.discriminator(bones.flatten(2))


class SANModel(nn.Module):
    """
    One `SANDomain` per motion domain, keyed by the dataset group name
    (e.g. "group_a", "group_b"). Domains share a latent space, so their skeletons must
    reduce to the same primal skeleton and have the same number of end-effectors.
    """

    def __init__(self, cfg: SANModelConfig, domain_specs: Dict[str, DomainSpec]):
        super().__init__()

        if len(domain_specs) != 2:
            raise ValueError(f"SAN trains exactly two domains, got: {list(domain_specs)}")

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
        """
        batch: {domain: {"motion": [B, T, 4E + 3], "bank_id": [B]}}

        Reconstructs every domain on its own banks, then runs all four source -> destination routes, each
        with a uniformly drawn target bank per sample, independently of how many windows a bank has. A
        same-domain route uses its drawn bank too, not the source's. Nothing is decoded back to the source:
        SAN has no motion cycle.
        """
        static = {name: domain.encode_offsets() for name, domain in self.domains.items()}

        motion_codes, reconstructions = {}, {}
        for name, domain in self.domains.items():
            source_static = select_banks(static[name], batch[name]["bank_id"])
            motion_codes[name] = domain.encode(batch[name]["motion"], source_static)
            reconstructions[name] = domain.decode(motion_codes[name], source_static)

        pairs = {}
        for source, motion_code in motion_codes.items():
            for destination, domain in self.domains.items():
                target_ids = torch.randint(domain.offsets.shape[0], (motion_code.shape[0],), device=motion_code.device)
                target_static = select_banks(static[destination], target_ids)

                fake = domain.decode(motion_code, target_static)
                pairs[(source, destination)] = PairOutputs(
                    target_ids=target_ids,
                    fake=fake,
                    fake_code=domain.encode(fake, target_static),
                )

        return GeneratorOutputs(motion_codes=motion_codes, reconstructions=reconstructions, pairs=pairs)

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.motion_encoder.parameters()
            yield from domain.offset_encoder.parameters()
            yield from domain.decoder.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.discriminator.parameters()

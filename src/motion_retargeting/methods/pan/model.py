from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterator

from torch import nn
import torch

from motion_retargeting.modules.decoders import MotionDecoder, ReferenceMotionDecoder
from motion_retargeting.modules.discriminators import MotionDiscriminator, ReferenceMotionDiscriminator
from motion_retargeting.modules.encoders import (
    BodyPartEncoder,
    MotionEncoder,
    ReferenceBodyPartEncoder,
    ReferenceMotionEncoder,
)
from motion_retargeting.data.utils import build_joint_mask

from .config import DomainSpec, PANModelConfig, PANReferenceDomainSpec, PANReferenceModelConfig
from .kinematics import EdgeForwardKinematics
from .masks import (
    attention_mask,
    parts_to_parts,
    parts_to_tokens,
    skeleton_offset_mask,
    temporal_conv_mask,
    tokens_to_parts,
)
from .utils import GeneratorOutputs, ReferenceGeneratorOutputs, ReferencePairOutputs


class PANDomain(nn.Module):
    """ Encoders / decoder / discriminator for a single skeleton domain. """

    def __init__(self, cfg: PANModelConfig, spec: DomainSpec):
        super().__init__()

        self.motion_encoder = MotionEncoder(
            input_dim=spec.rotation_dim,
            num_bodies=spec.num_bodies,
            joint_body_mask=build_joint_mask(spec.body_joints),
            **asdict(cfg.motion_encoder),
        )
        # Remove the root joint from the body part encoder input, on a copy: spec.body_joints is
        # shared with the datasets / topology (and the motion encoder above keeps the root).
        body_joints = {body: list(joints) for body, joints in spec.body_joints.items()}
        body_joints[spec.root_body].remove(0)
        self.body_part_encoder = BodyPartEncoder(
            input_dim=spec.offset_dim,
            body_joints=body_joints, # assumes root is joint 0
            **asdict(cfg.body_part_encoder),
        )
        self.decoder = MotionDecoder(
            input_dim=cfg.motion_encoder.embed_dim * spec.num_bodies,
            conv_out_dim=spec.motion_dim,
            **asdict(cfg.motion_decoder),
        )
        self.discriminator = MotionDiscriminator(
            input_dim=spec.motion_dim,
            **asdict(cfg.motion_discriminator),
        )


class PANModel(nn.Module):
    """
    One `PANDomain` per motion domain, keyed by the dataset group name
    (e.g. "group_a", "group_b").
    """

    def __init__(self, cfg: PANModelConfig, domain_specs: Dict[str, DomainSpec]):
        super().__init__()

        self.cfg = cfg
        self.specs = domain_specs
        self.domains = nn.ModuleDict(
            {name: PANDomain(cfg, spec) for name, spec in domain_specs.items()}
        )

    def forward_generator(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> GeneratorOutputs:
        """ batch: {domain: {"motion": [B, W, J+1, 4], "offsets": [B, J, 3]}} """
        domain_a = self.domains["group_a"]
        domain_b = self.domains["group_b"]

        motion_a, offsets_a = batch["group_a"]["motion"], batch["group_a"]["offsets"]
        motion_b, offsets_b = batch["group_b"]["motion"], batch["group_b"]["offsets"]

        # Encode
        z_motion_a = domain_a.motion_encoder(motion_a)
        z_motion_b = domain_b.motion_encoder(motion_b)

        z_skel_a = domain_a.body_part_encoder(offsets_a)
        z_skel_b = domain_b.body_part_encoder(offsets_b)

        z_skel_a = z_skel_a.unsqueeze(1).expand(z_motion_a.shape)
        z_skel_b = z_skel_b.unsqueeze(1).expand(z_motion_b.shape)

        # Self reconstruction
        group_aa = domain_a.decoder(z_motion_a + z_skel_a)
        group_bb = domain_b.decoder(z_motion_b + z_skel_b)

        # Cross-domain retargeting
        group_ab = domain_b.decoder(z_motion_a + z_skel_b)
        group_ba = domain_a.decoder(z_motion_b + z_skel_a)

        # Re-encode retargeted motions
        z_motion_ab = domain_b.motion_encoder(group_ab)
        z_motion_ba = domain_a.motion_encoder(group_ba)

        # Cycle reconstruction
        group_aba = domain_a.decoder(z_motion_ab + z_skel_a)
        group_bab = domain_b.decoder(z_motion_ba + z_skel_b)

        return GeneratorOutputs(
            group_aa=group_aa,
            group_bb=group_bb,
            group_ab=group_ab,
            group_ba=group_ba,
            group_aba=group_aba,
            group_bab=group_bab,
            motion_code_a=z_motion_a,
            motion_code_b=z_motion_b,
            motion_code_ab=z_motion_ab,
            motion_code_ba=z_motion_ba,
            skeleton_code_a=z_skel_a,
            skeleton_code_b=z_skel_b,
        )

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.motion_encoder.parameters()
            yield from domain.body_part_encoder.parameters()
            yield from domain.decoder.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.discriminator.parameters()


class PANReferenceDomain(nn.Module):
    """ Reference-recipe networks of one domain, with its masks and skeleton banks as persistent buffers. """

    def __init__(self, cfg: PANReferenceModelConfig, spec: PANReferenceDomainSpec):
        super().__init__()

        # A target's identity selects these four together, for same-domain retargets as well.
        self.register_buffer("offsets", spec.offsets)
        self.register_buffer("mean", spec.mean)
        self.register_buffer("std", spec.std)
        self.register_buffer("heights", spec.heights)

        self.fk = EdgeForwardKinematics(spec.parents.tolist())
        self.ee_ids = spec.ee_ids

        chains = spec.chains
        conv_mask = temporal_conv_mask(chains, spec.num_tokens)
        part_dim = cfg.motion_encoder.embed_dim

        part_connectivity = parts_to_parts(spec.num_parts, part_dim, part_dim)

        self.motion_encoder = ReferenceMotionEncoder(
            num_parts=spec.num_parts,
            attention_mask=attention_mask(chains, spec.num_tokens),
            shortcut_connectivity=tokens_to_parts(conv_mask, 4, part_dim),
            part_connectivity=part_connectivity,
            **asdict(cfg.motion_encoder),
        )
        self.body_part_encoder = ReferenceBodyPartEncoder(
            num_parts=spec.num_parts,
            input_connectivity=tokens_to_parts(skeleton_offset_mask(chains, spec.num_joints), 3, part_dim),
            part_connectivity=part_connectivity,
        )
        self.decoder = ReferenceMotionDecoder(
            part_connectivity=part_connectivity,
            output_connectivity=parts_to_tokens(conv_mask, part_dim, 4),
            **asdict(cfg.motion_decoder),
        )
        self.discriminator = ReferenceMotionDiscriminator(
            input_dim=spec.num_joints * 3,
            **asdict(cfg.motion_discriminator),
        )

    def denormalize(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ [B, T, 4E + 3], with each sample's bank statistics. """
        return motion * self.std[bank_ids][:, None] + self.mean[bank_ids][:, None]

    def bones(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Normalized motion -> bone_vectors_with_absolute_root on each sample's bank. """
        return self.fk.bone_vectors_with_absolute_root(self.denormalize(motion, bank_ids), self.offsets[bank_ids])


class PANReferenceModel(nn.Module):
    """ One `PANReferenceDomain` per motion domain, keyed by the dataset group name. """

    def __init__(self, cfg: PANReferenceModelConfig, domain_specs: Dict[str, PANReferenceDomainSpec]):
        super().__init__()

        self.cfg = cfg
        self.specs = domain_specs
        self.domains = nn.ModuleDict(
            {name: PANReferenceDomain(cfg, spec) for name, spec in domain_specs.items()}
        )

    def decode(self, domain: str, motion_code: torch.Tensor, skeleton_code: torch.Tensor) -> torch.Tensor:
        """ The only decoder entry: fuses z [B, T_ds, P, D] with s [B, P, D] as z + s / 100. """
        return self.domains[domain].decoder(motion_code + skeleton_code[:, None] / 100)

    def forward_generator(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> ReferenceGeneratorOutputs:
        """
        batch: {domain: {"motion": [B, T, 4E + 3], "bank_id": [B]}}

        Reconstructs every domain, then runs all four source -> destination routes, each with a
        uniformly drawn target bank per sample and independently of how many clips a bank has.
        """
        skeleton_codes = {
            name: domain.body_part_encoder(domain.offsets)
            for name, domain in self.domains.items()
        }

        motion_codes, reconstructions = {}, {}
        for name, domain in self.domains.items():
            motion_codes[name] = domain.motion_encoder(batch[name]["motion"])
            reconstructions[name] = self.decode(name, motion_codes[name], skeleton_codes[name][batch[name]["bank_id"]])

        pairs = {}
        for source in self.domains:
            source_codes = skeleton_codes[source][batch[source]["bank_id"]]
            motion_code = motion_codes[source]
            for destination, domain in self.domains.items():
                target_ids = torch.randint(domain.offsets.shape[0], (motion_code.shape[0],), device=motion_code.device)

                fake = self.decode(destination, motion_code, skeleton_codes[destination][target_ids])
                fake_code = domain.motion_encoder(fake)
                cycle = self.decode(source, fake_code, source_codes)

                pairs[(source, destination)] = ReferencePairOutputs(
                    target_ids=target_ids,
                    fake=fake,
                    fake_code=fake_code,
                    cycle=cycle,
                )

        return ReferenceGeneratorOutputs(motion_codes=motion_codes, reconstructions=reconstructions, pairs=pairs)

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.motion_encoder.parameters()
            yield from domain.body_part_encoder.parameters()
            yield from domain.decoder.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.discriminator.parameters()

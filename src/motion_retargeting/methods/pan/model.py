from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterator

from torch import nn
import torch

from motion_retargeting.modules.decoders import MotionDecoder
from motion_retargeting.modules.discriminators import MotionDiscriminator
from motion_retargeting.modules.encoders import BodyPartEncoder, MotionEncoder
from motion_retargeting.data.utils import build_joint_mask

from .config import DomainSpec, PANModelConfig
from .utils import GeneratorOutputs


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

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterator

from torch import nn
import torch

from motion_retargeting.modules.decoders import MotionDecoder
from motion_retargeting.modules.discriminators import MotionDiscriminator
from motion_retargeting.modules.encoders import BodyPartEncoder, MotionEncoder

from .config import DomainSpec, PANModelConfig
from .kinematics import JointForwardKinematics
from .masks import (
    attention_mask,
    parts_to_parts,
    parts_to_tokens,
    skeleton_offset_mask,
    temporal_conv_mask,
    tokens_to_parts,
)
from .utils import GeneratorOutputs, PairOutputs


class PANDomain(nn.Module):
    """ Networks of a single skeleton domain, with its masks and skeleton banks as persistent buffers. """

    def __init__(self, cfg: PANModelConfig, spec: DomainSpec):
        super().__init__()

        # A target's identity selects these four together, for same-domain retargets as well.
        self.register_buffer("offsets", spec.offsets)
        self.register_buffer("mean", spec.mean)
        self.register_buffer("std", spec.std)
        self.is_reference = spec.reference is not None
        if self.is_reference:
            reference = spec.reference
            self.register_buffer("heights", None)
            self.register_buffer("fk_offsets", reference.offsets_withend)
            self.fk = JointForwardKinematics(
                reference.parents_withend.tolist(), inclusive_velocity=True, not_end=reference.not_end.tolist()
            )
            self.ee_ids = []
        else:
            self.register_buffer("heights", spec.heights)
            self.fk = JointForwardKinematics(spec.parents.tolist())
            self.ee_ids = spec.ee_ids

        parts = spec.parts
        conv_mask = temporal_conv_mask(parts, spec.num_tokens)
        part_dim = cfg.motion_encoder.embed_dim

        part_connectivity = parts_to_parts(spec.num_parts, part_dim, part_dim)

        self.motion_encoder = MotionEncoder(
            num_parts=spec.num_parts,
            attention_mask=attention_mask(parts, spec.num_tokens),
            shortcut_connectivity=tokens_to_parts(conv_mask, 4, part_dim),
            part_connectivity=part_connectivity,
            **asdict(cfg.motion_encoder),
        )
        self.body_part_encoder = BodyPartEncoder(
            num_parts=spec.num_parts,
            input_connectivity=tokens_to_parts(skeleton_offset_mask(parts, spec.num_joints), 3, part_dim),
            part_connectivity=part_connectivity,
        )
        self.decoder = MotionDecoder(
            part_connectivity=part_connectivity,
            output_connectivity=parts_to_tokens(conv_mask, part_dim, 4),
            **asdict(cfg.motion_decoder),
        )
        self.discriminator = MotionDiscriminator(
            input_dim=len(self.fk.parents) * 3,
            raw_scores=self.is_reference,
            **asdict(cfg.motion_discriminator),
        )

    def denormalize(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ [B, T, 4J + 4], with each sample's bank statistics. """
        return motion * self.std[bank_ids][:, None] + self.mean[bank_ids][:, None]

    def bones(self, motion: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """ Normalized motion -> bone_vectors_with_absolute_root on each sample's bank. """
        offsets = self.fk_offsets if self.is_reference else self.offsets
        return self.fk.bone_vectors_with_absolute_root(self.denormalize(motion, bank_ids), offsets[bank_ids])

    def discriminator_input(self, bones: torch.Tensor) -> torch.Tensor:
        """Reference: all integrated positions in cm, including End Sites; legacy: bones."""
        return self.fk.world_positions(bones) if self.is_reference else bones


class PANModel(nn.Module):
    """
    One `PANDomain` per motion domain, keyed by the dataset group name
    (e.g. "group_a", "group_b").
    """

    def __init__(self, cfg: PANModelConfig, domain_specs: Dict[str, DomainSpec]):
        super().__init__()

        self.cfg = cfg
        self.specs = domain_specs
        reference_kinds = [spec.reference.kind for spec in domain_specs.values() if spec.reference is not None]
        self.is_reference = bool(reference_kinds)
        if self.is_reference and (len(domain_specs) != 2 or sorted(reference_kinds) != ["dog", "lafan1"]):
            raise ValueError("Reference PAN requires exactly one lafan1 and one dog archive domain")
        self.domains = nn.ModuleDict(
            {name: PANDomain(cfg, spec) for name, spec in domain_specs.items()}
        )

    def decode(self, domain: str, motion_code: torch.Tensor, skeleton_code: torch.Tensor) -> torch.Tensor:
        """ The only decoder entry: fuses z [B, T_ds, P, D] with s [B, P, D] as z + s / 100. """
        return self.domains[domain].decoder(motion_code + skeleton_code[:, None] / 100)

    def forward_generator(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> GeneratorOutputs:
        """
        batch: {domain: {"motion": [B, T, 4J + 4], "bank_id": [B]}}

        Reconstructs every domain. Legacy: all four routes with uniformly drawn target banks.
        Reference archives: exactly two cross routes using each destination batch's banks.
        """
        if self.is_reference and (
            set(batch) != set(self.domains) or len({item["motion"].shape[0] for item in batch.values()}) != 1
        ):
            raise ValueError("Reference cross routes require equally sized destination batches")
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
                if self.is_reference:
                    if source == destination:
                        continue
                    target_ids = batch[destination]["bank_id"]
                else:
                    target_ids = torch.randint(domain.offsets.shape[0], (motion_code.shape[0],), device=motion_code.device)

                fake = self.decode(destination, motion_code, skeleton_codes[destination][target_ids])
                fake_code = domain.motion_encoder(fake)
                cycle = self.decode(source, fake_code, source_codes)

                pairs[(source, destination)] = PairOutputs(
                    target_ids=target_ids,
                    fake=fake,
                    fake_code=fake_code,
                    cycle=cycle,
                )

        return GeneratorOutputs(motion_codes=motion_codes, reconstructions=reconstructions, pairs=pairs)

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.motion_encoder.parameters()
            yield from domain.body_part_encoder.parameters()
            yield from domain.decoder.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        for domain in self.domains.values():
            yield from domain.discriminator.parameters()

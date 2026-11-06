"""
SAN reference recipe's switch, domain spec and config, and PAN's data view through them, on synthetic clips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan.data import ReferenceDataConfig, build_from_collections, split_holdout
from motion_retargeting.methods.san.config import (
    REFERENCE_RECIPE,
    SANConfig,
    SANModelConfig,
    SANReferenceConfig,
    SANReferenceDomainSpec,
)

ROOT = Path(__file__).resolve().parents[1]

# The AMASS and CMU skeletons of configs/methods/san_reference.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]


def clip(num_joints: int, scale: float = 1.0, seed: int = 0, frames: int = 130) -> MotionData:
    generator = torch.Generator().manual_seed(seed)
    rotations = torch.nn.functional.normalize(torch.randn(frames, num_joints, 4, generator=generator), dim=-1)
    offsets = scale * 0.1 * torch.ones(num_joints, 3)
    offsets[0] = 0.0
    root = torch.randn(frames, 3, generator=generator)
    return MotionData(offsets=offsets, joint_rotations=rotations, fps=30.0, root_position=root)


def dataset_for(parents, scales=(1.0, 1.0, 1.5)):
    collection = MotionCollection(
        topology=SkeletonTopology(parents=torch.tensor(parents)),
        motions=[clip(len(parents), scale, seed) for seed, scale in enumerate(scales)],
    )
    return build_from_collections([collection], ReferenceDataConfig())


class TestRecipeSwitch:
    def test_legacy_config_defaults_to_the_legacy_recipe(self):
        assert SANConfig().recipe == "legacy"
        assert SANConfig(recipe="legacy").recipe == "legacy"

    @pytest.mark.parametrize("recipe", [REFERENCE_RECIPE, "other"])
    def test_legacy_config_rejects_every_other_recipe(self, recipe):
        with pytest.raises(ValueError, match="Unknown SAN recipe"):
            SANConfig(recipe=recipe)

    def test_reference_config_only_runs_its_recipe(self):
        assert SANReferenceConfig().recipe == "mixamo_reference"
        with pytest.raises(ValueError, match="only runs recipe"):
            SANReferenceConfig(recipe="legacy")

    def test_reference_weights(self):
        cfg = SANReferenceConfig()
        assert (cfg.replay_buffer_size, cfg.swap_probability) == (50, 0.5)
        assert (cfg.reconstruction_lambda, cfg.root_lambda, cfg.position_lambda) == (5, 1250, 500)
        assert (cfg.latent_lambda, cfg.end_effector_lambda, cfg.contact_lambda) == (2.5, 50, 100)
        assert (cfg.contact_threshold, cfg.adversarial_lambda) == (0.008, 1)


class TestYaml:
    def load(self, name: str) -> dict:
        return yaml.safe_load((ROOT / "configs" / "methods" / name).read_text())

    def test_model_blocks_are_legacy_san_blocks(self):
        reference, legacy = self.load("san_reference.yaml"), self.load("san.yaml")
        assert reference["method"]["name"] == "san"
        for section in ("motion_encoder", "offset_encoder", "motion_decoder"):
            assert reference["method"]["model"][section] == legacy["method"]["model"][section]
        for block in ("block1", "block2"):
            assert (
                reference["method"]["model"]["motion_discriminator"][block]
                == legacy["method"]["model"]["motion_discriminator"][block]
            )
        SANModelConfig.from_dict(reference["method"]["model"])

    def test_params_build_the_reference_config(self):
        cfg = self.load("san_reference.yaml")
        assert SANReferenceConfig(**cfg["method"]["params"]).recipe == REFERENCE_RECIPE

    def test_trainer_is_the_reference_step_budget(self):
        trainer = self.load("san_reference.yaml")["trainer"]
        assert (trainer["batch_size"], trainer["learning_rate"], trainer["num_epochs"]) == (256, 2e-4, 1539)
        assert 1539 * 104 >= 160_008 > 1538 * 104

    def test_dataset_groups_are_the_reference_view(self):
        for group in self.load("san_reference.yaml")["dataset"].values():
            assert ReferenceDataConfig(**group["params"]) == ReferenceDataConfig()


class TestDomainSpec:
    @pytest.mark.parametrize(
        "parents, ee_ids",
        [(AMASS_PARENTS, [10, 11, 15, 20, 21]), (CMU_PARENTS, [4, 8, 15, 19, 23])],
    )
    def test_end_effectors_are_the_chain_endpoints(self, parents, ee_ids):
        spec = SANReferenceDomainSpec.from_dataset(dataset_for(parents))
        assert spec.ee_ids == ee_ids

    def test_sizes_come_from_the_dataset(self):
        spec = SANReferenceDomainSpec.from_dataset(dataset_for(CMU_PARENTS))
        assert spec.num_banks == 2
        assert spec.offsets.shape == (2, 24, 3)
        assert spec.mean.shape == spec.std.shape == (2, 95)
        assert len(spec.chains) == 6

    def test_heights_are_left_foot_plus_head_paths_not_the_longest_chain(self):
        spec = SANReferenceDomainSpec.from_dataset(dataset_for(AMASS_PARENTS, scales=(1.0, 2.0)))
        bone = 0.1 * 3 ** 0.5

        # AMASS: left foot tip 10 is 4 bones from the root, head tip 15 is 5 bones.
        assert spec.heights == pytest.approx(torch.tensor([9 * bone, 18 * bone]), rel=1e-5)
        # The legacy height is the longest root-to-joint chain, the 7 bones to a hand.
        assert (spec.heights > 7 * bone).all()

    def test_heights_follow_the_path_not_the_vertical_extent(self):
        spec = SANReferenceDomainSpec.from_dataset(dataset_for(CMU_PARENTS, scales=(1.0,)))
        bone = 0.1 * 3 ** 0.5
        # CMU: left foot tip 4 is 4 bones from the root, head tip 15 is 7 bones.
        assert spec.heights.item() == pytest.approx(11 * bone, rel=1e-5)

    def test_other_topologies_are_rejected_when_the_spec_is_built(self):
        extra_chain = AMASS_PARENTS + [0, 22]
        with pytest.raises(ValueError, match="6 body parts"):
            SANReferenceDomainSpec(
                parents=torch.tensor(extra_chain),
                offsets=torch.zeros(1, 24, 3),
                mean=torch.zeros(1, 95),
                std=torch.ones(1, 95),
            )


class TestBatches:
    @pytest.mark.parametrize("parents, channels", [(AMASS_PARENTS, 87), (CMU_PARENTS, 95)])
    def test_batch_holds_edge_motion_and_bank_ids(self, parents, channels):
        train, holdout = split_holdout(dataset_for(parents))
        batch = next(iter(torch.utils.data.DataLoader(train, batch_size=2)))

        assert batch["motion"].shape == (2, 64, channels)
        assert batch["bank_id"].shape == (2,)
        assert set(batch) == {"motion", "bank_id"}
        assert train.augment and not holdout.augment

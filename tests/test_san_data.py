"""
SAN's domain spec and config, and its data view through them, on synthetic clips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.san.data import (
    SANDataConfig,
    build_from_collections,
    resample_indices,
    split_holdout,
    to_edge_channels,
)
from motion_retargeting.methods.san.config import DomainSpec, SANConfig, SANModelConfig

ROOT = Path(__file__).resolve().parents[1]

# The AMASS and CMU skeletons of configs/methods/san.yaml.
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
    return build_from_collections([collection], SANDataConfig())


class TestConfig:
    def test_defaults_are_the_official_weights(self):
        cfg = SANConfig()
        assert (cfg.replay_buffer_size, cfg.swap_probability) == (50, 0.5)
        assert tuple(cfg.betas) == (0.9, 0.999)
        assert cfg.reconstruction_lambda == 5
        assert cfg.reconstruction_lambda * cfg.position_lambda == 500
        assert cfg.reconstruction_lambda * cfg.position_lambda * cfg.root_position_lambda == 1250
        assert (cfg.cycle_consistency_lambda, cfg.end_effector_lambda, cfg.contact_lambda) == (2.5, 50, 100)
        assert (cfg.contact_velocity_threshold, cfg.adversarial_lambda) == (0.008, 1)


class TestYaml:
    def load(self, name: str) -> dict:
        return yaml.safe_load((ROOT / "configs" / "methods" / name).read_text())

    def test_model_blocks_are_the_default_model_config(self):
        cfg = self.load("san.yaml")
        assert cfg["method"]["name"] == "san"
        assert SANModelConfig.from_dict(cfg["method"]["model"]) == SANModelConfig()

    def test_params_are_the_default_config(self):
        cfg = SANConfig(**self.load("san.yaml")["method"]["params"])
        cfg.betas = tuple(cfg.betas)
        cfg.loss_grad_norm_interval = 0
        assert cfg == SANConfig()

    def test_trainer_is_the_official_step_budget(self):
        trainer = self.load("san.yaml")["trainer"]
        assert (trainer["batch_size"], trainer["learning_rate"], trainer["num_epochs"]) == (256, 2e-4, 1539)
        assert 1539 * 104 >= 160_008 > 1538 * 104

    def test_dataset_groups_are_the_official_view(self):
        for group in self.load("san.yaml")["dataset"].values():
            assert SANDataConfig(**group["params"]) == SANDataConfig()


class TestDomainSpec:
    @pytest.mark.parametrize(
        "parents, end_effectors",
        [(AMASS_PARENTS, [10, 11, 15, 20, 21]), (CMU_PARENTS, [4, 8, 15, 19, 23])],
    )
    def test_end_effectors_are_the_chain_endpoints(self, parents, end_effectors):
        spec = DomainSpec.from_dataset(dataset_for(parents))
        assert spec.end_effectors == end_effectors

    def test_sizes_come_from_the_dataset(self):
        spec = DomainSpec.from_dataset(dataset_for(CMU_PARENTS))
        assert spec.num_banks == 2
        assert spec.offsets.shape == (2, 24, 3)
        assert spec.mean.shape == spec.std.shape == (2, 95)
        assert len(spec.chains) == 6

    def test_heights_are_left_foot_plus_head_paths_not_the_longest_chain(self):
        spec = DomainSpec.from_dataset(dataset_for(AMASS_PARENTS, scales=(1.0, 2.0)))
        bone = 0.1 * 3 ** 0.5

        # AMASS: left foot tip 10 is 4 bones from the root, head tip 15 is 5 bones.
        assert spec.heights == pytest.approx(torch.tensor([9 * bone, 18 * bone]), rel=1e-5)

    def test_heights_follow_the_path_not_the_vertical_extent(self):
        spec = DomainSpec.from_dataset(dataset_for(CMU_PARENTS, scales=(1.0,)))
        bone = 0.1 * 3 ** 0.5
        # CMU: left foot tip 4 is 4 bones from the root, head tip 15 is 7 bones.
        assert spec.heights.item() == pytest.approx(11 * bone, rel=1e-5)

    def test_other_topologies_are_rejected_when_the_spec_is_built(self):
        extra_chain = AMASS_PARENTS + [0, 22]
        with pytest.raises(ValueError, match="6 body parts"):
            DomainSpec(
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


# The root branches into 1 -> 2 and 3; joints 2 and 3 are leaves.
BRANCHING_PARENTS = torch.tensor([-1, 0, 1, 0])


def branching_clip(offset_scale: float = 1.0, num_frames: int = 128, fps: float = 30.0, seed: int = 0) -> MotionData:
    generator = torch.Generator().manual_seed(seed)
    rotations = torch.nn.functional.normalize(torch.randn(num_frames, 4, 4, generator=generator), dim=-1)
    offsets = offset_scale * torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.2, 0.0, 0.0]])
    root = torch.randn(num_frames, 3, generator=generator) + torch.tensor([3.0, 1.0, -2.0])
    return MotionData(offsets=offsets, joint_rotations=rotations, fps=fps, root_position=root)


def branching_collection(*clips: MotionData) -> MotionCollection:
    return MotionCollection(topology=SkeletonTopology(parents=BRANCHING_PARENTS), motions=list(clips))


class TestEdgeChannels:
    def test_edges_carry_parent_rotation_and_absolute_root(self):
        data = branching_clip(num_frames=8)
        channels = to_edge_channels(data.joint_rotations, data.root_position, BRANCHING_PARENTS)

        assert channels.shape == (8, 4 * 3 + 3)
        # Root rotation repeats on both outgoing edges, joint 1's on its only one.
        assert torch.equal(channels[:, 0:4], data.joint_rotations[:, 0])
        assert torch.equal(channels[:, 4:8], data.joint_rotations[:, 1])
        assert torch.equal(channels[:, 8:12], data.joint_rotations[:, 0])
        # Nonzero root height / translation survive, untouched.
        assert torch.equal(channels[:, -3:], data.root_position)

    def test_leaf_rotations_are_excluded(self):
        data = branching_clip(num_frames=8)
        channels = to_edge_channels(data.joint_rotations, data.root_position, BRANCHING_PARENTS)

        changed = data.joint_rotations.clone()
        changed[:, [2, 3]] = 1.0
        assert torch.equal(channels, to_edge_channels(changed, data.root_position, BRANCHING_PARENTS))

    def test_no_sign_repair_or_heading_removal(self):
        data = branching_clip(num_frames=8)
        flipped = data.joint_rotations.clone()
        flipped[3, 0] = -flipped[3, 0]
        channels = to_edge_channels(flipped, data.root_position, BRANCHING_PARENTS)
        assert torch.equal(channels[3, :4], flipped[3, 0])


class TestResampling:
    @pytest.mark.parametrize("fps, step", [(120.0, 4), (60.0, 2), (30.0, 1)])
    def test_integer_ratios_have_fixed_steps(self, fps, step):
        indices = resample_indices(int(fps) * 3 + 1, fps, 30.0)
        assert (indices[1:] - indices[:-1] == step).all()
        assert indices[0] == 0

    def test_stops_before_the_clip_duration(self):
        assert resample_indices(120, 120.0, 30.0).tolist() == list(range(0, 120, 4))
        assert resample_indices(121, 120.0, 30.0).tolist() == list(range(0, 121, 4))

    def test_noninteger_ratio_picks_the_nearest_frame(self):
        # Timestamps k / 30 s at 50 Hz fall on source frames k * 5 / 3.
        assert resample_indices(10, 50.0, 30.0).tolist() == [0, 2, 3, 5, 7, 8]
        # 120 / 1.2 lands just above 100 in float64: no timestamp may be added for it.
        assert resample_indices(120, 36.0, 30.0).shape[0] == 100


class TestDataset:
    def test_window_count_matches_official_arithmetic(self):
        for num_frames in (64, 65, 95, 96, 128, 200):
            dataset = build_from_collections([branching_collection(branching_clip(num_frames=num_frames))], SANDataConfig())
            assert len(dataset) == num_frames // 32 - 1

    def test_too_short_clips_make_no_bank(self):
        dataset = build_from_collections(
            [branching_collection(branching_clip(offset_scale=2.0, num_frames=10), branching_clip(num_frames=64))],
            SANDataConfig(),
        )
        assert dataset.num_banks == 1
        assert torch.equal(dataset.offsets[0], branching_clip().offsets)

    def test_banks_by_exact_offset_equality_in_first_occurrence_order(self):
        almost_one = 1.0 + 1e-6
        dataset = build_from_collections(
            [branching_collection(branching_clip(1.0), branching_clip(2.0), branching_clip(1.0, seed=1), branching_clip(almost_one))],
            SANDataConfig(),
        )
        assert dataset.num_banks == 3
        assert dataset.bank_ids.unique_consecutive().tolist() == [0, 1, 0, 2]

    def test_singleton_bank_is_legal(self):
        dataset = build_from_collections([branching_collection(branching_clip())], SANDataConfig())
        assert dataset.num_banks == 1

    def test_normalize_denormalize_round_trip_on_all_channels(self):
        dataset = build_from_collections([branching_collection(branching_clip(1.0), branching_clip(2.0, seed=1))], SANDataConfig())

        for index in (0, len(dataset) - 1):
            item = dataset[index]
            bank = item["bank_id"]
            denormalized = item["motion"] * dataset.std[bank] + dataset.mean[bank]
            assert torch.allclose(denormalized, dataset.motion[index], atol=1e-5)

        # Rotation channels are normalized too.
        window = torch.stack([dataset[i]["motion"] for i in range(len(dataset)) if dataset.bank_ids[i] == 0])
        assert torch.allclose(window.mean(dim=(0, 1)), torch.zeros(15), atol=1e-4)
        assert torch.allclose(window.std(dim=(0, 1)), torch.ones(15), atol=1e-3)

    def test_constant_channels_stay_finite(self):
        data = branching_clip()
        data.joint_rotations[...] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        dataset = build_from_collections([branching_collection(data)], SANDataConfig())

        assert (dataset.std[0, :12] == 1.0).all()
        assert torch.isfinite(dataset[0]["motion"]).all()

    def test_different_banks_use_different_statistics(self):
        dataset = build_from_collections([branching_collection(branching_clip(1.0), branching_clip(2.0, seed=1))], SANDataConfig())
        assert not torch.equal(dataset.mean[0], dataset.mean[1])

        index = (dataset.bank_ids == 1).nonzero()[0].item()
        expected = (dataset.motion[index] - dataset.mean[1]) / dataset.std[1]
        assert torch.equal(dataset[index]["motion"], expected)


class TestHoldout:
    def make_dataset(self):
        return build_from_collections(
            [branching_collection(branching_clip(1.0, num_frames=1024), branching_clip(2.0, num_frames=512, seed=1))],
            SANDataConfig(),
        )

    def test_split_is_95_5_in_order_per_bank(self):
        dataset = self.make_dataset()
        train, holdout = split_holdout(dataset, 5)

        for bank in range(dataset.num_banks):
            windows = dataset.motion[dataset.bank_ids == bank]
            num_train = windows.shape[0] * 95 // 100
            assert torch.equal(train.motion[train.bank_ids == bank], windows[:num_train])
            assert torch.equal(holdout.motion[holdout.bank_ids == bank], windows[num_train:])

    def test_statistics_are_fit_before_the_holdout(self):
        dataset = self.make_dataset()
        train, holdout = split_holdout(dataset, 5)
        assert train.mean is dataset.mean and holdout.mean is dataset.mean
        assert train.std is dataset.std and holdout.std is dataset.std

    def test_only_training_samples_reverse_in_time(self):
        dataset = self.make_dataset()
        train, holdout = split_holdout(dataset, 5)

        torch.manual_seed(0)
        forward = train.normalize(train.motion[0], train.bank_ids[0])
        draws = [train[0]["motion"] for _ in range(200)]
        reversed_draws = sum(torch.equal(draw, forward.flip(0)) for draw in draws)
        assert sum(torch.equal(draw, forward) for draw in draws) + reversed_draws == 200
        assert 60 < reversed_draws < 140

        expected = holdout.normalize(holdout.motion[0], holdout.bank_ids[0])
        assert all(torch.equal(holdout[0]["motion"], expected) for _ in range(50))

    def test_topology_mismatch_is_rejected(self):
        other = MotionCollection(topology=SkeletonTopology(parents=torch.tensor([-1, 0, 0, 0])), motions=[branching_clip()])
        with pytest.raises(ValueError):
            build_from_collections([branching_collection(branching_clip()), other], SANDataConfig())

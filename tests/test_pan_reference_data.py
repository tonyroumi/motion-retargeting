"""
Edge / absolute-root data view of the PAN reference recipe, on synthetic clips.
"""

from __future__ import annotations

import pytest
import torch

from motion_retargeting.data.builder import MotionDatasetBuilder, MotionDatasetBuilderConfig
from motion_retargeting.data.dataset import MotionDataset, compute_stats
from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan.data import (
    ReferenceDataConfig,
    build_from_collections,
    resample_indices,
    split_holdout,
    to_edge_channels,
)

# The root branches into 1 -> 2 and 3; joints 2 and 3 are leaves.
BRANCHING_PARENTS = torch.tensor([-1, 0, 1, 0])


def clip(offset_scale: float = 1.0, num_frames: int = 128, fps: float = 30.0, seed: int = 0) -> MotionData:
    generator = torch.Generator().manual_seed(seed)
    rotations = torch.nn.functional.normalize(torch.randn(num_frames, 4, 4, generator=generator), dim=-1)
    offsets = offset_scale * torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.2, 0.0, 0.0]])
    root = torch.randn(num_frames, 3, generator=generator) + torch.tensor([3.0, 1.0, -2.0])
    return MotionData(offsets=offsets, joint_rotations=rotations, fps=fps, root_position=root)


def collection(*clips: MotionData) -> MotionCollection:
    return MotionCollection(topology=SkeletonTopology(parents=BRANCHING_PARENTS), motions=list(clips))


class TestEdgeChannels:
    def test_edges_carry_parent_rotation_and_absolute_root(self):
        data = clip(num_frames=8)
        channels = to_edge_channels(data.joint_rotations, data.root_position, BRANCHING_PARENTS)

        assert channels.shape == (8, 4 * 3 + 3)
        # Root rotation repeats on both outgoing edges, joint 1's on its only one.
        assert torch.equal(channels[:, 0:4], data.joint_rotations[:, 0])
        assert torch.equal(channels[:, 4:8], data.joint_rotations[:, 1])
        assert torch.equal(channels[:, 8:12], data.joint_rotations[:, 0])
        # Nonzero root height / translation survive, untouched.
        assert torch.equal(channels[:, -3:], data.root_position)

    def test_leaf_rotations_are_excluded(self):
        data = clip(num_frames=8)
        channels = to_edge_channels(data.joint_rotations, data.root_position, BRANCHING_PARENTS)

        changed = data.joint_rotations.clone()
        changed[:, [2, 3]] = 1.0
        assert torch.equal(channels, to_edge_channels(changed, data.root_position, BRANCHING_PARENTS))

    def test_no_sign_repair_or_heading_removal(self):
        data = clip(num_frames=8)
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
    def test_window_count_matches_reference_arithmetic(self):
        for num_frames in (64, 65, 95, 96, 128, 200):
            dataset = build_from_collections([collection(clip(num_frames=num_frames))], ReferenceDataConfig())
            assert len(dataset) == num_frames // 32 - 1

    def test_too_short_clips_make_no_bank(self):
        dataset = build_from_collections(
            [collection(clip(offset_scale=2.0, num_frames=10), clip(num_frames=64))],
            ReferenceDataConfig(),
        )
        assert dataset.num_banks == 1
        assert torch.equal(dataset.offsets[0], clip().offsets)

    def test_banks_by_exact_offset_equality_in_first_occurrence_order(self):
        almost_one = 1.0 + 1e-6
        dataset = build_from_collections(
            [collection(clip(1.0), clip(2.0), clip(1.0, seed=1), clip(almost_one))],
            ReferenceDataConfig(),
        )
        assert dataset.num_banks == 3
        assert dataset.bank_ids.unique_consecutive().tolist() == [0, 1, 0, 2]

    def test_singleton_bank_is_legal(self):
        dataset = build_from_collections([collection(clip())], ReferenceDataConfig())
        assert dataset.num_banks == 1

    def test_normalize_denormalize_round_trip_on_all_channels(self):
        dataset = build_from_collections([collection(clip(1.0), clip(2.0, seed=1))], ReferenceDataConfig())

        for index in (0, len(dataset) - 1):
            item = dataset[index]
            bank = item["bank_id"]
            denormalized = item["motion"] * dataset.std[bank] + dataset.mean[bank]
            assert torch.allclose(denormalized, dataset.motion[index], atol=1e-5)

        # Rotation channels are normalized too, unlike the legacy final-row-only tensor.
        window = torch.stack([dataset[i]["motion"] for i in range(len(dataset)) if dataset.bank_ids[i] == 0])
        assert torch.allclose(window.mean(dim=(0, 1)), torch.zeros(15), atol=1e-4)
        assert torch.allclose(window.std(dim=(0, 1)), torch.ones(15), atol=1e-3)

    def test_constant_channels_stay_finite(self):
        data = clip()
        data.joint_rotations[...] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        dataset = build_from_collections([collection(data)], ReferenceDataConfig())

        assert (dataset.std[0, :12] == 1.0).all()
        assert torch.isfinite(dataset[0]["motion"]).all()

    def test_different_banks_use_different_statistics(self):
        dataset = build_from_collections([collection(clip(1.0), clip(2.0, seed=1))], ReferenceDataConfig())
        assert not torch.equal(dataset.mean[0], dataset.mean[1])

        index = (dataset.bank_ids == 1).nonzero()[0].item()
        expected = (dataset.motion[index] - dataset.mean[1]) / dataset.std[1]
        assert torch.equal(dataset[index]["motion"], expected)


class TestHoldout:
    def make_dataset(self):
        return build_from_collections(
            [collection(clip(1.0, num_frames=1024), clip(2.0, num_frames=512, seed=1))],
            ReferenceDataConfig(),
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
        other = MotionCollection(topology=SkeletonTopology(parents=torch.tensor([-1, 0, 0, 0])), motions=[clip()])
        with pytest.raises(ValueError):
            build_from_collections([collection(clip()), other], ReferenceDataConfig())


class TestLegacyPathUntouched:
    def test_legacy_windows_keep_final_row_only_normalization(self, tmp_path, monkeypatch):
        """ The legacy builder still produces the [W, J+1, 4] window, normalizing only its final row. """
        from types import SimpleNamespace

        from motion_retargeting.data.adapters import MotionSourceAdapter, base

        monkeypatch.setattr(base, "PATHS", SimpleNamespace(data_root=tmp_path))
        monkeypatch.setattr(base, "label_joint_bodies", lambda topology, offsets: {0: list(range(topology.num_joints))})

        class Adapter(MotionSourceAdapter):
            def _extract(self):
                data = clip()
                return [self._construct_motion(
                    parents=BRANCHING_PARENTS,
                    offsets=data.offsets,
                    joint_rotations=data.joint_rotations,
                    root_position=data.root_position,
                    fps=data.fps,
                )]

        builder = MotionDatasetBuilder(MotionDatasetBuilderConfig(sequence_length=64, stride=64))
        dataset = builder.build([Adapter("legacy")])
        assert isinstance(dataset, MotionDataset)
        assert dataset.motion.shape[1:] == (64, 5, 4)

        raw = dataset.motion[0].clone()
        dataset.set_normalization(*compute_stats(dataset))
        item = dataset[0]["motion"]
        assert torch.equal(item[:, :-1], raw[:, :-1])
        assert not torch.equal(item[:, -1], raw[:, -1])

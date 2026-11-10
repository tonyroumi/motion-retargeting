"""PAN `max_train_windows`: disabled, enabled and boundary caps, applied after the holdout split."""

import pytest
import torch

from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan import data as pan_data
from motion_retargeting.methods.pan.data import (
    PANDataConfig,
    build_from_collections,
    build_pan_splits,
    limit_windows,
    split_holdout,
)

PARENTS = torch.tensor([-1, 0, 1])
PARTS = {0: [0], 1: [1, 2]}
PARAMS = {"sequence_length": 8, "stride": 4, "target_fps": None, "holdout_percent": 20}


def collection(seed: int = 0) -> MotionCollection:
    generator = torch.Generator().manual_seed(seed)
    clips = []
    for bank, frames in enumerate([45, 61, 33]):  # three skeleton banks
        offsets = torch.tensor([[0., 0., 0.], [0., .3 + .1 * bank, 0.], [0., .2, 0.]])
        rotations = torch.nn.functional.normalize(torch.randn(frames, 3, 4, generator=generator), dim=-1)
        root = torch.randn(frames, 3, generator=generator).cumsum(0) * .01
        clips.append(MotionData(offsets, rotations, 30., root))
    return MotionCollection(SkeletonTopology(PARENTS, body_joints=PARTS), clips)


def train_split():
    dataset = build_from_collections([collection()], PANDataConfig(**PARAMS))
    return split_holdout(dataset, PARAMS["holdout_percent"])[0]


def rows(dataset):
    return {tuple(window.flatten().tolist()) for window in dataset.motion}


def test_disabled_keeps_every_window():
    train = train_split()
    assert limit_windows(train, None) is train
    assert PANDataConfig().max_train_windows is None


@pytest.mark.parametrize("extra", [0, 1, 1000])
def test_cap_at_or_above_size_is_a_no_op(extra):
    train = train_split()
    assert limit_windows(train, len(train) + extra) is train


@pytest.mark.parametrize("cap", [1, 5])
def test_cap_selects_a_seeded_ordered_subset(cap):
    train = train_split()
    limited = limit_windows(train, cap, seed=3)
    assert len(limited) == cap
    assert rows(limited) <= rows(train)
    assert limited.offsets is train.offsets and limited.mean is train.mean and limited.std is train.std

    # Temporal order: the selected windows appear in the same order as in `train`.
    positions = [next(i for i in range(len(train)) if torch.equal(train.motion[i], w)) for w in limited.motion]
    assert positions == sorted(positions)

    again = limit_windows(train, cap, seed=3)
    assert torch.equal(again.motion, limited.motion) and torch.equal(again.bank_ids, limited.bank_ids)


def test_seed_changes_the_selection():
    train = train_split()
    motions = [limit_windows(train, 5, seed=seed).motion for seed in range(4)]
    assert any(not torch.equal(motions[0], m) for m in motions[1:])


@pytest.mark.parametrize("cap", [0, -1, True, 2.5])
def test_invalid_caps_raise(cap):
    with pytest.raises(ValueError, match="max_train_windows"):
        limit_windows(train_split(), cap)


def test_build_splits_caps_only_training_and_never_draws_from_holdout(monkeypatch):
    class Adapter:
        def __init__(self, name):
            self.seed = {"a": 0, "b": 1}[name]

        def load(self):
            return collection(self.seed)

    monkeypatch.setattr(pan_data, "build_adapter", Adapter)

    def config(cap=None):
        return {
            "group_a": {"sources": ["a"], "params": {**PARAMS, "max_train_windows": cap}},
            "group_b": {"sources": ["b"], "params": dict(PARAMS)},
        }

    full_train, full_holdout = build_pan_splits(config())
    train, holdout = build_pan_splits(config(cap=4))

    assert len(train["group_a"]) == 4
    assert rows(train["group_a"]) <= rows(full_train["group_a"])
    assert torch.equal(train["group_b"].motion, full_train["group_b"].motion)  # other domain untouched
    for name in holdout:
        assert torch.equal(holdout[name].motion, full_holdout[name].motion)
        assert torch.equal(holdout[name].bank_ids, full_holdout[name].bank_ids)
        assert torch.equal(train[name].mean, full_train[name].mean)
        assert torch.equal(train[name].offsets, full_train[name].offsets)

"""
PAN's training-step safety: non-finite gradients stop a step before either optimizer touches the weights,
generator clipping bounds large finite gradients, and a step's capture replays it and identifies its batch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import PANMotionDataset, fit_statistics, rebuild_batch
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.utils import NonFiniteGradientError

# Root 0 with three children; joint 3 with three more: one joint per labeled part beyond the root's.
PARENTS = [-1, 0, 0, 0, 3, 3, 3]
BODY_JOINTS = {0: [0, 1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6]}
OFFSETS = torch.tensor(
    [[0.0, 0.0, 0.0], [0.1, -0.5, 0.0], [-0.1, -0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0], [0.3, 0.2, 0.0], [-0.3, 0.2, 0.0]]
)
NUM_EDGES = len(PARENTS) - 1
NUM_WINDOWS = 8
NUM_FRAMES = 16
BATCH_SIZE = 4
DOMAINS = ("a", "b")


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


def make_dataset(scale: float) -> PANMotionDataset:
    """ Random windows over two banks, with a known (source, clip, start) per window. """
    rotations = torch.nn.functional.normalize(
        torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.2 * torch.randn(NUM_WINDOWS, NUM_FRAMES, NUM_EDGES, 4), dim=-1
    )
    motion = torch.cat([rotations.flatten(2), 0.02 * torch.randn(NUM_WINDOWS, NUM_FRAMES, 4)], dim=-1)
    bank_ids = torch.arange(NUM_WINDOWS) % 2
    mean, std = fit_statistics(motion, bank_ids, 2)
    index = torch.arange(NUM_WINDOWS)
    windows = torch.stack([torch.zeros_like(index), index // 2, 32 * (index % 2)], dim=-1)
    return PANMotionDataset(
        motion=motion,
        bank_ids=bank_ids,
        offsets=torch.stack([OFFSETS * scale, OFFSETS * scale * 1.1]),
        mean=mean,
        std=std,
        parents=torch.tensor(PARENTS),
        body_joints=BODY_JOINTS,
        augment=True,
        windows=windows,
    )


def make_method(datasets: dict, **cfg) -> PANMethod:
    specs = {name: DomainSpec.from_dataset(dataset) for name, dataset in datasets.items()}
    # A contact threshold every generated velocity is under, so the contact term always has a selection.
    method = PANMethod(PANModelConfig(), specs, cfg=PANConfig(contact_threshold=1e9, **cfg))
    method.train()
    return method


def batch_of(datasets: dict, indices: list[int]) -> dict:
    return {name: default_collate([dataset[i] for i in indices]) for name, dataset in datasets.items()}


@pytest.fixture
def datasets() -> dict:
    return {"a": make_dataset(1.0), "b": make_dataset(1.3)}


def generator_parameter(method: PANMethod, domain: str = "a") -> torch.nn.Parameter:
    return next(method.model.domains[domain].decoder.parameters())


def discriminator_parameter(method: PANMethod, domain: str = "a") -> torch.nn.Parameter:
    return next(method.model.domains[domain].discriminator.parameters())


def weights(method: PANMethod) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in method.model.parameters()]


def optimizer_moments(method: PANMethod) -> list[torch.Tensor]:
    return [
        state[key].clone()
        for optimizer in (method.generator_optimizer, method.discriminator_optimizer)
        for state in optimizer.state.values()
        for key in ("exp_avg", "exp_avg_sq")
    ]


def all_equal(first: list[torch.Tensor], second: list[torch.Tensor]) -> bool:
    return len(first) == len(second) and all(torch.equal(x, y) for x, y in zip(first, second))


class TestNonFiniteGradients:
    """ After a normal step, so Adam has state that a NaN step would also poison. """

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    @pytest.mark.parametrize("network", ["generator", "discriminator"])
    def test_neither_optimizer_steps(self, datasets, tmp_path, value, network):
        method = make_method(datasets)
        method.capture_dir = tmp_path
        method.train_step(batch_of(datasets, [0, 1, 2, 3]))
        before, moments = weights(method), optimizer_moments(method)

        parameter = generator_parameter(method) if network == "generator" else discriminator_parameter(method)
        parameter.register_hook(lambda grad: grad * value)
        with pytest.raises(NonFiniteGradientError) as error:
            method.train_step(batch_of(datasets, [4, 5, 6, 7]))

        assert all_equal(weights(method), before)
        assert all_equal(optimizer_moments(method), moments)
        assert all(torch.isfinite(w).all() for w in weights(method))

        component = "decoder" if network == "generator" else "discriminator"
        assert error.value.parameters == [f"a/{component}/" + next(getattr(method.model.domains["a"], component).named_parameters())[0]]

        (capture_path,) = tmp_path.glob("step_*_failure.pt")
        capture = torch.load(capture_path, weights_only=False)
        assert capture["report"]["nonfinite_gradient_parameters"] == error.value.parameters
        assert capture["num_steps"] == 1


class TestGeneratorClipping:
    def run(self, datasets, max_grad_norm: float) -> tuple[dict, float]:
        method = make_method(datasets, generator_max_grad_norm=max_grad_norm)
        generator_parameter(method).register_hook(lambda grad: grad * 1e6)
        metrics = method.train_step(batch_of(datasets, [0, 1, 2, 3]))
        applied = torch.linalg.vector_norm(
            torch.stack([p.grad.norm() for p in method.model.generator_parameters() if p.grad is not None])
        )
        return metrics, applied.item()

    def test_bounds_a_large_finite_gradient(self, datasets):
        metrics, applied = self.run(datasets, max_grad_norm=1.0)
        assert metrics["grad/generator"] > 1e3  # logged before clipping
        assert applied == pytest.approx(1.0, rel=1e-4)

    def test_off_leaves_the_gradient(self, datasets):
        metrics, applied = self.run(datasets, max_grad_norm=0.0)
        assert applied == pytest.approx(metrics["grad/generator"], rel=1e-5)


class TestSpikeDetection:
    def test_flags_a_jump_over_the_baseline_and_follows_a_lasting_rise(self, datasets):
        method = make_method(datasets, grad_spike_factor=10.0)
        assert all(method.spike_bound(5.0) is None for _ in range(50))
        assert method.spike_bound(500.0) == pytest.approx(50.0)
        assert method.grad_baseline < 6.0  # one spike barely moves the baseline
        flagged = [method.spike_bound(500.0) is not None for _ in range(100)]
        assert flagged[0] and not flagged[-1]


def spiking_method(datasets: dict, tmp_path: Path, **cfg) -> PANMethod:
    """ Every step after the first, which sets the baseline, is a spike. """
    method = make_method(datasets, grad_spike_factor=1e-12, **cfg)
    method.capture_dir = tmp_path
    method.train_step(batch_of(datasets, [0, 2, 4, 6]))
    return method


class TestCapture:
    def test_replays_the_step(self, datasets, tmp_path):
        # A replay buffer small enough to fill and start swapping, so its contents and RNG matter.
        method = spiking_method(datasets, tmp_path, spike_capture_limit=2, replay_buffer_size=3)
        steps = [[0, 1, 2, 3], [4, 5, 6, 7], [1, 3, 5, 7]]
        for indices in steps[:2]:
            method.train_step(batch_of(datasets, indices))
        expected = method.train_step(batch_of(datasets, steps[2]))
        expected_weights = weights(method)

        # Only the latest two spike captures are kept.
        assert sorted(path.name for path in tmp_path.iterdir()) == ["step_00000003_spike.pt", "step_00000004_spike.pt"]
        capture = torch.load(tmp_path / "step_00000004_spike.pt", weights_only=False)
        assert all("motion" not in item for item in capture["batch"].values())  # no training data by default

        torch.manual_seed(123)  # Different initial weights and RNG, all replaced by the capture
        replay = make_method(datasets, replay_buffer_size=3)
        replay.restore_capture(capture)
        assert replay.train_step(rebuild_batch(datasets, capture["batch"])) == expected
        assert all_equal(weights(replay), expected_weights)

    def test_identifies_the_outlying_window(self, datasets, tmp_path):
        method = spiking_method(datasets, tmp_path)
        datasets["b"].motion[5, 3, 2] += 50.0  # one window, far outside its bank's statistics
        method.train_step(batch_of(datasets, [4, 5, 6, 7]))

        report = torch.load(tmp_path / "step_00000002_spike.pt", weights_only=False)["report"]
        largest = report["domains"]["b"]["largest_inputs"][0]
        assert largest["index"] == 5
        assert largest["window"] == datasets["b"].windows[5].tolist()
        assert largest["bank"] == 1
        assert largest["bank_std"] == pytest.approx(datasets["b"].std[1, largest["channel"]].item())
        assert set(largest["target_banks"]) == set(DOMAINS)

    def test_a_changed_dataset_is_refused(self, datasets, tmp_path):
        method = spiking_method(datasets, tmp_path)
        method.train_step(batch_of(datasets, [0, 1, 2, 3]))
        capture = torch.load(tmp_path / "step_00000002_spike.pt", weights_only=False)

        datasets["a"].motion[0] += 0.1
        with pytest.raises(ValueError, match="do not match"):
            rebuild_batch(datasets, capture["batch"])

    def test_stores_the_motion_without_dataset_indices(self, datasets, tmp_path):
        method = spiking_method(datasets, tmp_path)
        batch = batch_of(datasets, [0, 1, 2, 3])
        batch = {name: {"motion": item["motion"], "bank_id": item["bank_id"]} for name, item in batch.items()}
        method.train_step(batch)

        capture = torch.load(tmp_path / "step_00000002_spike.pt", weights_only=False)
        assert capture["report"]["domains"]["a"]["provenance"].startswith("unavailable")
        rebuilt = rebuild_batch(datasets, capture["batch"])
        assert torch.equal(rebuilt["a"]["motion"], batch["a"]["motion"])

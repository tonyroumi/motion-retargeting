"""
PAN's "<domain>/root" generator term: the reconstruction's root trajectory, integrated from its root velocity
as FK does, against the source's, over the source's height. Not a per-frame velocity error.
"""

from __future__ import annotations

import pytest
import torch

from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.method import PANMethod
from motion_retargeting.methods.pan.utils import ForwardState, GeneratorOutputs, SourceState

# Root 0 with three children; joint 3 with three more: one joint per labeled part beyond the root's.
PARENTS = [-1, 0, 0, 0, 3, 3, 3]
BODY_JOINTS = {0: [0, 1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6]}
OFFSETS = torch.tensor(
    [[0.0, 0.0, 0.0], [0.1, -0.5, 0.0], [-0.1, -0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0], [0.3, 0.2, 0.0], [-0.3, 0.2, 0.0]]
)
NUM_CHANNELS = 4 * (len(PARENTS) - 1) + 4
NUM_FRAMES = 8
ROOT_XYZ = slice(-4, -1)
YAW = -1


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


@pytest.fixture
def method() -> PANMethod:
    """ Identity statistics, so the channels are already physical. """
    spec = DomainSpec(
        parents=torch.tensor(PARENTS),
        offsets=OFFSETS[None],
        mean=torch.zeros(1, NUM_CHANNELS),
        std=torch.ones(1, NUM_CHANNELS),
        body_joints=BODY_JOINTS,
    )
    return PANMethod(PANModelConfig(), {"a": spec}, cfg=PANConfig())


def source_motion() -> torch.Tensor:
    """ [1, T, 4E + 4]: identity edge rotations, a random root velocity and yaw rate. """
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, NUM_FRAMES, len(PARENTS) - 1)
    return torch.cat([rotations, torch.randn(1, NUM_FRAMES, 4)], dim=-1)


def root_loss(method: PANMethod, motion: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    domain = method.model.domains["a"]
    bank_ids = torch.zeros(motion.shape[0], dtype=torch.long)
    source = SourceState(
        bank_ids=bank_ids,
        motion=motion,
        denormalized=domain.denormalize(motion, bank_ids),
        bones=domain.bones(motion, bank_ids).detach(),
        reconstruction=reconstruction,
        reconstruction_denormalized=domain.denormalize(reconstruction, bank_ids),
        reconstruction_bones=domain.bones(reconstruction, bank_ids),
    )
    state = ForwardState(outputs=GeneratorOutputs({}, {}, {}), sources={"a": source}, pairs={})
    return method.compute_generator_losses(state)["a/root"]


def height(method: PANMethod) -> float:
    return method.model.domains["a"].heights[0].item()


def with_velocity_error(motion: torch.Tensor, frame: int, error: torch.Tensor) -> torch.Tensor:
    reconstruction = motion.clone()
    reconstruction[:, frame, ROOT_XYZ] += error
    return reconstruction


def test_matches_the_integrated_trajectory_error(method):
    """ Frame t's root is the sum of the steps before it, so the window starts at the origin. """
    motion = source_motion()
    reconstruction = motion + 0.1 * torch.randn_like(motion)

    def trajectory(channels):
        velocity = channels[..., ROOT_XYZ]
        return torch.cumsum(velocity, dim=1) - velocity

    expected = method.cfg.root_lambda * torch.mean(
        ((trajectory(reconstruction) - trajectory(motion)) / height(method)) ** 2
    )
    assert torch.allclose(root_loss(method, motion, reconstruction), expected, rtol=1e-5)


def test_an_early_velocity_error_persists_in_every_later_frame(method):
    """ A per-frame velocity loss would score these equally; the trajectory error lasts T - 1 - k frames. """
    motion = source_motion()
    error = torch.tensor([0.3, -0.2, 0.1])
    scale = method.cfg.root_lambda * error.pow(2).sum().item() / (3 * NUM_FRAMES * height(method) ** 2)

    for frame in (0, 3, NUM_FRAMES - 2):
        loss = root_loss(method, motion, with_velocity_error(motion, frame, error))
        assert loss.item() == pytest.approx(scale * (NUM_FRAMES - 1 - frame), rel=1e-5)


def test_the_last_step_moves_no_frame_of_the_window(method):
    """ The last frame's velocity is the step out of the window, so no frame's root depends on it. """
    motion = source_motion()
    reconstruction = with_velocity_error(motion, NUM_FRAMES - 1, torch.tensor([0.3, -0.2, 0.1]))
    assert root_loss(method, motion, reconstruction).item() == pytest.approx(0.0, abs=1e-10)


def test_cancelling_velocity_errors_only_cost_the_frames_between_them(method):
    """ +e at step 2 and -e at step 5: the trajectory is off at frames 3, 4, 5 only. """
    motion = source_motion()
    error = torch.tensor([0.3, -0.2, 0.1])
    reconstruction = with_velocity_error(with_velocity_error(motion, 2, error), 5, -error)
    expected = method.cfg.root_lambda * 3 * error.pow(2).sum().item() / (3 * NUM_FRAMES * height(method) ** 2)
    assert root_loss(method, motion, reconstruction).item() == pytest.approx(expected, rel=1e-5)


def test_ignores_the_yaw_rate(method):
    motion = source_motion()
    reconstruction = motion.clone()
    reconstruction[..., YAW] += 1.0
    assert root_loss(method, motion, reconstruction).item() == pytest.approx(0.0, abs=1e-10)


def test_gradient_reaches_the_reconstruction(method):
    motion = source_motion()
    reconstruction = with_velocity_error(motion, 0, torch.tensor([0.3, -0.2, 0.1])).requires_grad_(True)
    root_loss(method, motion, reconstruction).backward()
    assert reconstruction.grad[0, 0, ROOT_XYZ].abs().sum() > 0
    assert reconstruction.grad[0, -1, ROOT_XYZ].abs().sum() == 0

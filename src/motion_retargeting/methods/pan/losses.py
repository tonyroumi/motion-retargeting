import torch
import torch.nn.functional as F


def generator_adversarial_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """ Fake patch scores target 1, averaged over squared patch errors rather than scores. """
    return F.mse_loss(fake_scores, torch.ones_like(fake_scores))


def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """ L_D = 0.5 * (mse(D(real), 1) + mse(D(fake), 0)), over patch scores. """
    real_loss = F.mse_loss(real_scores, torch.ones_like(real_scores))
    fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    return 0.5 * (real_loss + fake_loss)


def mapped_root_velocity(velocity: torch.Tensor, minimum: float, maximum: float, zero_safe: bool = False) -> torch.Tensor:
    """Reference speed-range mapping, preserving direction (no clamp/extrapolation limit).

    Zero speed is undefined in the reference; preserve that singularity rather than
    silently change the objective. The training finite-loss check rejects it.
    The explicitly experimental human protocol maps exact zero to zero instead;
    all nonzero speeds still use the reference formula (no epsilon/clamping).
    """
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    denominator = torch.where(speed == 0, torch.ones_like(speed), speed) if zero_safe else speed
    return ((speed - minimum) / (maximum - minimum)) * velocity / denominator


def end_effector_terms(
    source_velocity: torch.Tensor,
    fake_velocity: torch.Tensor,
    contact_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (transfer, contact) MSEs between normalized end-effector velocities [B, T - 1, 5, 3].

    Contact frames are where the *source* end effector is planted (velocity norm below the threshold),
    and there the generated velocity is pulled toward the source's. This departs from the official code,
    which selects them on the generated velocity: that mask drains itself as the generator moves the
    selected frames toward a moving source, until a batch has none and the official term turns NaN.

    A source batch with no planted end effector has nothing to hold still, so its contact term is zero,
    kept in the graph so per-term gradients stay defined.
    """
    contacts = torch.linalg.vector_norm(source_velocity, dim=-1) < contact_threshold
    transfer = F.mse_loss(fake_velocity, source_velocity)
    if not contacts.any():
        return transfer, transfer * 0.0
    return transfer, F.mse_loss(fake_velocity[contacts], source_velocity[contacts])

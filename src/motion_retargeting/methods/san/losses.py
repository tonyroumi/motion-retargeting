"""
SAN training objectives.
"""

import torch
import torch.nn.functional as F


def adversarial_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """ Least-squares discriminator loss: real motion is pushed to 1, retargeted motion to 0. """
    real_loss = F.mse_loss(real_scores, torch.ones_like(real_scores))
    fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))

    return 0.5 * (real_loss + fake_loss)


def generator_adversarial_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """ Least-squares generator loss: retargeted motion is pushed to 1. """
    return F.mse_loss(fake_scores, torch.ones_like(fake_scores))


def end_effector_terms(
    source_velocity: torch.Tensor,
    fake_velocity: torch.Tensor,
    velocity_threshold: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (transfer, contact) MSEs between normalized end-effector velocities [B, T - 1, E, 3]. As in the official
    code, contact frames are selected on the *generated* velocity, not the source's.

    An empty selection is NaN in the official code, which then trains through it. Here it raises instead:
    a zero contact term would be a different objective, not this one.
    """
    contacts = torch.linalg.vector_norm(fake_velocity, dim=-1) < velocity_threshold
    if not contacts.any():
        raise RuntimeError(
            f"End-effector loss: no generated velocity is below the contact threshold {velocity_threshold} "
            f"(the official contact term is NaN here). Stopping before any optimizer step of this batch."
        )
    return F.mse_loss(fake_velocity, source_velocity), F.mse_loss(fake_velocity[contacts], source_velocity[contacts])

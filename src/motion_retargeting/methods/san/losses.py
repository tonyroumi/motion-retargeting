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


def end_effector_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    contact_lambda: float = 100.0,
    velocity_threshold: float = 0.008,
) -> torch.Tensor:
    """
    End-effector velocity loss over [B, W-1, E, 3], with an extra term on the end-effectors
    that are in contact (moving slower than `velocity_threshold`) in the target.
    """
    loss = F.mse_loss(predicted, target)

    contact = target.norm(dim=-1) < velocity_threshold
    if contact.any():
        loss = loss + contact_lambda * F.mse_loss(predicted[contact], target[contact])

    return loss

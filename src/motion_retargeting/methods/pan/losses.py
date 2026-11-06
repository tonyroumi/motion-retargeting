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


def end_effector_terms(
    source_velocity: torch.Tensor,
    fake_velocity: torch.Tensor,
    contact_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (transfer, contact) MSEs between normalized end-effector velocities [B, T - 1, 5, 3]. As in the
    official code at its pinned commit, contact frames are selected on the *generated* velocity, not
    the source's.

    An empty selection is NaN in the official code, which then trains through it. Here it raises
    instead: a zero contact term would be a different objective, not this one.
    """
    contacts = torch.linalg.vector_norm(fake_velocity, dim=-1) < contact_threshold
    if not contacts.any():
        raise RuntimeError(
            f"End-effector loss: no generated velocity is below the contact threshold "
            f"{contact_threshold} (the official contact term is NaN here). Stopping before any "
            f"optimizer step of this batch."
        )
    return F.mse_loss(fake_velocity, source_velocity), F.mse_loss(fake_velocity[contacts], source_velocity[contacts])

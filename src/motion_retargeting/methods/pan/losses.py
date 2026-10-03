import torch
import torch.nn.functional as F


def adversarial_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    L_adv = ||C^B(M_T^B_real)||^2
          + ||1 - C^B(M_T^{A->B})||^2
    """

    real_loss = torch.mean(real_logits ** 2)
    fake_loss = torch.mean((1.0 - fake_logits) ** 2)

    return real_loss + fake_loss

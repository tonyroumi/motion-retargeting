import torch
import torch.nn as nn
import torch.nn.functional as F

class LossBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.lsgan_loss = LSGANLoss()
        self.ee_loss = EELoss()

    def mse(self, pred, gt):
        return F.mse_loss(pred, gt)

    def mae(self, pred, gt):
        return F.l1_loss(pred, gt)

    def lsgan(self, *, d_args=None, g_args=None):
        if d_args is not None:
            return self.lsgan_loss(d_real=d_args, d_fake=g_args, is_discriminator=True)

        return self.lsgan_loss(d_fake=g_args, is_discriminator=False)

    def ee(self, pred, gt):
        return self.ee_loss(pred, gt)

class LSGANLoss(nn.Module):
    """
    Least Squares GAN loss.

    is_discriminator = True:
        real -> 1
        fake -> 0

    is_discriminator = False (generator):
        fake -> 1
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        *,
        d_real: torch.Tensor | None = None,
        d_fake: torch.Tensor | None = None,
        is_discriminator: bool,
    ) -> torch.Tensor:

        if is_discriminator:
            real_targets = torch.ones_like(d_real)
            fake_targets = torch.zeros_like(d_fake)

            loss_real = self.mse(d_real, real_targets)
            loss_fake = self.mse(d_fake, fake_targets)

            return 0.5 * (loss_real + loss_fake)

        else:
            real_targets = torch.ones_like(d_fake)
            return self.mse(d_fake, real_targets)

class EELoss(nn.Module):
    """ End-Effector velocity loss. """
    def __init__(self, norm_eps=0.008):
        super().__init__()
        self.mse = nn.MSELoss()
        self.norm_eps = norm_eps
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ee_loss = self.mse(pred, gt)
 
        gt_norm = torch.norm(gt, dim=-1)
        contact_idx = gt_norm < self.norm_eps
        extra_ee_loss = self.mse(pred[contact_idx], gt[contact_idx])
 
        return ee_loss + extra_ee_loss * 100
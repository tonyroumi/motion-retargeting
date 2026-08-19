"""
Trainer for an unpaired skeletal motion GAN.
"""
from .losses import LossBundle

from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple, List
import torch

from src.core.types import MotionOutput, PairedSample
from src.models.networks.gan import SkeletalGAN
from src.utils import Logger, ImagePool

class SkeletalGANTrainer:
    def __init__(
        self,
        model: SkeletalGAN,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        train_loader: DataLoader,
        checkpoint_dir: str,
        config: DictConfig,
        logger: Logger,
        device: torch.device = 'cpu',
    ):
        self.model : SkeletalGAN = model.to(device)

        self.losses = LossBundle()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

        self.train_loader = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.device = device
        self.logger = logger

        self.image_pools : List[ImagePool] = [
            ImagePool(config.buffer_size) 
            for _ in range(len(self.model.topologies))
        ]

        self.logger.info(f"Trainer and model Initialized on: {device}")
    
    def train(self) -> None:
        total_loss = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            
            epoch_loss = self._train_one_epoch()
            
            total_loss += epoch_loss

            self.logger.log_metric("loss/total_loss", total_loss/(epoch+1))

            self.logger.epoch()

            if (epoch % self.config.checkpoint_interval == 0):
                self._save_checkpoint(epoch)
    
    def _train_one_epoch(self) -> Any:
        total_epoch_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)

            rec_outputs, ret_outputs = self.model(batch)

            # Generator
            self.model.discriminators_requires_grad_(False)
            self.optimizer_G.zero_grad()
            generator_loss = self._backward_G(rec_outputs, ret_outputs, batch=batch)
            self.optimizer_G.step()

            # Discriminator
            self.model.discriminators_requires_grad_(True)
            self.optimizer_D.zero_grad()
            discriminator_loss = self._backward_D(ret_outputs, original_world_pos=batch.gt_positions)
            self.optimizer_D.step()

            total_epoch_loss += (generator_loss + discriminator_loss)
    
            self.logger.step()

        return total_epoch_loss

    def _backward_D(
        self,
        ret_outputs: Dict[Tuple[int, int], MotionOutput],
        original_world_pos: Tuple[torch.Tensor, torch.Tensor] 
    ) -> torch.Tensor:
        """
        A->A, A->B, B->A, B->B
        """
        tot_D_loss = 0.0
        first = ret_outputs[1,0].positions.flatten(start_dim=-2)
        second = ret_outputs[0,1].positions.flatten(start_dim=-2)
        both = [first, second]
        for i in range(2):
            pred_fake = self.model.forward_discriminator(
                self.image_pools[i].query(both[i]).detach(),
                idx=i
            )

            pred_real = self.model.forward_discriminator(
                original_world_pos[i].flatten(start_dim=-2),
                idx=i
            )

            loss_D = self.losses.lsgan(d_args=pred_real, g_args=pred_fake)
            self.logger.log_metric(f"loss/D_loss_{i}", loss_D)
            loss_D.backward()

            tot_D_loss += loss_D

        # for (src, dst), out in ret_outputs.items(): #TODO(anthony) unsure if we want to do this for each cross section. only want A->A, A->B. sum like that 
        #     if src==dst:
        #         continue

        #     pred_fake = self.model.forward_discriminator(
        #         self.image_pools[src].query(out.positions.flatten(start_dim=-2)).detach(),
        #         idx=src
        #     )

        #     pred_real = self.model.forward_discriminator(
        #         original_world_pos[src].flatten(start_dim=-2),
        #         idx=src
        #     )

        #     loss_D = self.losses.lsgan(d_args=pred_real, g_args=pred_fake)
        #     self.logger.log_metric(f"loss/D_loss_{src}", loss_D)
        #     loss_D.backward()

        #     tot_D_loss += loss_D

        return tot_D_loss       

    def _backward_G(
        self, 
        rec_outputs: Dict[int, MotionOutput],
        ret_outputs: Dict[Tuple[int, int], MotionOutput],
        batch: PairedSample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # -----------------------
        # Reconstruction loss
        # -----------------------
        rec_loss = 0
        for i, out in rec_outputs.items():
            rec_motion_loss = self.losses.mse(pred=out.motion, gt=batch.motions[i])
            self.logger.log_metric(f"loss/rec_motion_{i}", rec_motion_loss)

            original_root_pos = batch.rotations[i][:, -3:] / batch.heights[i][:, None, None]
            rec_root_pos = out.rotations[:, -3:] / batch.heights[i][:, None, None]
            rec_root_pos_loss = self.losses.mse(
                pred=rec_root_pos,
                gt=original_root_pos
            ) 
            self.logger.log_metric(f"loss/rec_root_pos_{i}", value=rec_root_pos_loss)

            original_world_pos = batch.gt_positions[i] / batch.heights[i][:, None, None, None]
            rec_world_pos = out.positions / batch.heights[i][:, None, None, None]
            rec_joint_pos_loss = self.losses.mse(pred=rec_world_pos, gt=original_world_pos)
            self.logger.log_metric(f"loss/rec_joint_pos_{i}", value=rec_joint_pos_loss)

            rec_loss += rec_motion_loss + (rec_root_pos_loss * 2.5 + rec_joint_pos_loss) * 100

        # -----------------------
        # Retargetting loss
        # -----------------------
        tot_cycle_loss, tot_ee_loss, tot_G_loss = 0, 0, 0
        for (src, dst), out in ret_outputs.items():
            cycle_loss = self.losses.mae(pred=rec_outputs[src].latents, gt=out.latents)
            self.logger.log_metric(f"loss/cycle_{src}->{dst}", value=cycle_loss)
            tot_cycle_loss += cycle_loss

            ee_loss = self.losses.ee(pred=out.ee_vels, gt=batch.gt_ee_vels[src])
            self.logger.log_metric(f"loss/ee_vel_{src}->{dst}", value=ee_loss)
            tot_ee_loss += ee_loss

            if src != dst:
                G_loss = self.losses.lsgan(
                    g_args=self.model.forward_discriminator(out.positions.flatten(start_dim=-2), dst)
                )
                self.logger.log_metric(f"loss/G_loss_{src}->{dst}", value=G_loss)
                tot_G_loss += G_loss
        
        total_G_loss = rec_loss * 5 + \
                       tot_cycle_loss * 2.5 + \
                       tot_ee_loss + 50 * \
                       tot_G_loss
        total_G_loss.backward()

        return total_G_loss

    def _save_checkpoint(self, epoch: int) -> None:
        """ Saves all states and necessary data to resume training and load model for inference """
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            'topologies': self.model.topologies,
            'normalization_stats': tuple(d.norm_stats for d in self.model.domains),
            "config" : {
                "offset_encoder" : self.model.offset_encoder_params,
                "auto_encoder": self.model.auto_encoder_params,
                "discriminator": self.model.discriminator_params
            }
        }

        path = ckpt_dir / f"skeletal_gan_epoch{epoch:03d}.pt"
        torch.save(state, path)

        self.logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """ Load trainer from checkpoint  """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        try:
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("Model state loaded successfully")
            
            # Load optimizer states
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            self.logger.info("Generator optimizer state loaded successfully")
            
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
            self.logger.info("Discriminator optimizer state loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
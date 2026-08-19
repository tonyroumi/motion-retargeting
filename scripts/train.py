"""
SkeletalGAN Training Script

Train a motion retargeting network between source and target skeletal structures.
"""

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional
import argparse
import hydra
import torch

from src.data.adapters import get_adapter_for_character
from src.data.datasets import (
    CrossDomainMotionDataset,
    MotionDataset,
    paired_collate
)
from src.models.networks import SkeletalGAN
from src.training import SkeletalGANTrainer
from src.utils import set_seed, Logger, SkeletonVisualizer

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
   
    logger = Logger(log_dir=str(output_dir), verbose=cfg.verbose)
       
    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")
       
    dataset_A = MotionDataset(character=cfg.source, device=cfg.device)
    dataset_B = MotionDataset(character=cfg.target, device=cfg.device)
    
    logger.info(f"Dataset A: {len(dataset_A)} samples ({cfg.source})")
    logger.info(f"Dataset B: {len(dataset_B)} samples ({cfg.target})")
    
    cross_domain_dataset = CrossDomainMotionDataset(dataset_A, dataset_B)
    
    train_loader = torch.utils.data.DataLoader(
        cross_domain_dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=paired_collate,
        shuffle=True,
    )
    
    # Visualize data if requested
    if cfg.visualize_data_first:
        logger.info("Visualizing dataset samples...")
        SkeletonVisualizer.visualize_dataset(
            train_loader,
            save_path=str(output_dir / 'data_visualization')
        )
    
    model = SkeletalGAN(
        topologies=cross_domain_dataset.topologies,
        normalization_stats=(dataset_A.norm_stats, dataset_B.norm_stats),
        offset_encoder_params=cfg.model.offset_encoder,
        auto_encoder_params=cfg.model.autoencoder,
        discriminator_params=cfg.model.discriminator
    )
    
    # Create optimizers
    optimizer_G = torch.optim.Adam(
        model.generator_parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.betas,
    )
    
    optimizer_D = torch.optim.Adam(
        model.discriminator_parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.betas,
    )
    
    # Initialize trainer
    trainer = SkeletalGANTrainer(
        model=model,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        train_loader=train_loader,
        checkpoint_dir=str(output_dir),
        config=cfg.train,
        logger=logger,
        device=cfg.device
    )
    
    if cfg.resume_from:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from}")
        trainer.load_checkpoint(cfg.resume_from)
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
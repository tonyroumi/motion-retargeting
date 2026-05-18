from dataclasses import dataclass
from moveitmoveit.src.types import BaseParams

@dataclass(frozen=True)
class PPOHyperparams(BaseParams):
    num_mini_batches: int = 1
    num_learning_epochs: int = 4
    clip_param: float = 0.2
    discount: float = 0.97
    td_lambda: float = 0.95
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False

    value_loss_coef: float = 1.0
    entropy_coef: float = 0.2

@dataclass(frozen=True)
class AMPHyperparams(PPOHyperparams):
    # Discriminator replay-buffer settings
    discriminator_buffer_capacity: int = 100_000
    disc_replay_samples: int = 1000
    # Discriminator optimizer
    disc_lr: float = 1e-4
    disc_logit_reg: float = 0.01
    disc_weight_decay: float = 0.0001

    # How often (in policy update iterations) to update the discriminator
    discriminator_update_interval: int = 4

    # Number of gradient steps per discriminator update
    disc_num_updates: int = 5
    disc_batch_size: int = 2048

    disc_grad_penalty_coef: float = 5.0
    num_disc_obs_steps: int = 10

    disc_reward_lambda: float = 2 
    goal_reward_lambda: float = 1.0
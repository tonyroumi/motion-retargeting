from __future__ import annotations

import torch
import torch.nn as nn

from moveitmoveit.src.buffers import CircularObsBuffer
from moveitmoveit.src.models import AMPNetworks
from utils import Logger

from .ppo import PPO
from .params import AMPHyperparams

class AMP(PPO):
    """Adversarial Motion Priors (AMP) algorithm. """
    networks: AMPNetworks
    params: AMPHyperparams

    def __init__(
        self,
        networks: AMPNetworks,
        params: AMPHyperparams,
        logger: Logger,
    ):  
        super().__init__(networks, params, logger)
        
        self._discriminator_update_count = 0

    def init_storage(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        super().init_storage(num_envs, num_transitions, obs_dim, action_dim)

        self.discriminator_storage = CircularObsBuffer(
            obs_dim=obs_dim,
            disc_obs_steps=self.params.num_disc_obs_steps,
            n_envs=num_envs,
            capacity=self.params.discriminator_buffer_capacity,
            device=self.networks.device
        )
        self.discriminator_ref_storage = CircularObsBuffer(
            obs_dim=obs_dim,
            disc_obs_steps=self.params.num_disc_obs_steps,
            n_envs=num_envs,
            capacity=self.params.discriminator_buffer_capacity,
            device=self.networks.device
        )

    def process_reset(self, infos: dict, env_ids=None) -> None:
        """Seed the discriminator sliding window from motion-lib frames returned at reset."""
        self.discriminator_storage.seed_from_windows(infos["disc_obs"])

    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:

        num_samples = infos["disc_obs"].shape[0]
        rand_idx = torch.randperm(num_samples, device=self.networks.device, dtype=torch.long)

        if (self.discriminator_storage.is_full):
            num_samples = min(num_samples, self.params.disc_replay_samples) 
        
        idx = rand_idx[:num_samples]

        stacked_disc_obs = self.discriminator_storage.add(infos["disc_obs"][idx])
        ref_disc_obs = self.discriminator_ref_storage.add(infos["ref_disc_obs"][idx])

        self.networks.record_disc_obs(stacked_disc_obs)
        self.networks.record_disc_obs(ref_disc_obs)

        normed_disc_obs = self.networks.normalize_disc_obs(stacked_disc_obs)

        with torch.no_grad():
            d = self.networks.disc(normed_disc_obs).squeeze()
        
        prob = torch.sigmoid(d)
        disc_reward = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
        goal_reward = rewards.clone()
        rewards = self.params.disc_reward_lambda * disc_reward.detach() + self.params.goal_reward_lambda * goal_reward

        self.logger.log_metric("reward/disc_mean", disc_reward.mean().item())
        self.logger.log_metric("reward/goal_mean", goal_reward.mean().item())
        self.logger.log_metric("reward/disc_std", disc_reward.std().item())
        self.logger.log_metric("reward/goal_std", goal_reward.std().item())

        super().process_env_step(rewards, terminated, truncated, infos)

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        super().update(optimizer)

        if self._update_count % self.params.discriminator_update_interval == 0:
            self._update_discriminator(optimizer)

    def _update_discriminator(self, optimizer: torch.optim.Optimizer) -> None:
        """Run one round of discriminator gradient updates."""

        mean_disc_loss = 0.0
        mean_disc_ref_accuracy = 0.0
        mean_disc_agent_accuracy = 0.0
        mean_disc_ref_logits = 0.0
        mean_disc_agent_logits = 0.0

        for _ in range(self.params.disc_num_updates):
            agent_input = self.discriminator_storage.sample(self.params.disc_batch_size, stacked=False)
            normed_agent_input = self.networks.disc_obs_norm.normalize(agent_input)

            ref_input = self.discriminator_ref_storage.sample(self.params.disc_batch_size, stacked=False)
            normed_ref_input = self.networks.disc_obs_norm.normalize(ref_input)
            normed_ref_input.requires_grad_(True)

            agent_logits = self.networks.disc(normed_agent_input)
            ref_logits = self.networks.disc(normed_ref_input)

            agent_loss = nn.functional.binary_cross_entropy_with_logits(
                agent_logits, torch.zeros_like(agent_logits)
            )
            ref_loss = nn.functional.binary_cross_entropy_with_logits(
                ref_logits, torch.ones_like(ref_logits)
            )
            disc_loss = 0.5 * (agent_loss + ref_loss)

            logit_weights = self.networks.discriminator.get_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self.params.disc_logit_reg * disc_logit_loss

            disc_weight_decay = sum(
                p.pow(2).sum() for p in self.networks.discriminator.parameters()
            )
            disc_loss += self.params.disc_weight_decay * disc_weight_decay

            # gradient penalty
            disc_ref_grad = torch.autograd.grad(
                outputs=ref_logits,
                inputs=normed_ref_input,
                grad_outputs=torch.ones_like(ref_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Square and sum the gradients and take the mean.
            disc_ref_grad_norm = disc_ref_grad.pow(2).sum(dim=-1).mean()
            disc_loss = disc_loss + self.params.disc_grad_penalty_coef * disc_ref_grad_norm

            optimizer.zero_grad()
            disc_loss.backward()
            optimizer.step()

            agent_accuracy = torch.mean((agent_logits < 0).float())
            ref_disc_accuracy = torch.mean((ref_logits > 0).float()) 

            mean_disc_loss += disc_loss.item()
            mean_disc_ref_accuracy += ref_disc_accuracy.item()
            mean_disc_agent_accuracy += agent_accuracy.item()
            mean_disc_ref_logits += torch.mean(ref_logits).item()
            mean_disc_agent_logits += torch.mean(agent_logits).item()

        mean_disc_loss /= self.params.disc_num_updates
        mean_disc_ref_accuracy /= self.params.disc_num_updates
        mean_disc_agent_accuracy /= self.params.disc_num_updates
        mean_disc_ref_logits /= self.params.disc_num_updates
        mean_disc_agent_logits /= self.params.disc_num_updates

        self.logger.log_metric("disc/loss", mean_disc_loss)
        self.logger.log_metric("disc/agent_accuracy", mean_disc_agent_accuracy)
        self.logger.log_metric("disc/ref_accuracy", mean_disc_ref_accuracy)
        self.logger.log_metric("disc/agent_logits", mean_disc_agent_logits)
        self.logger.log_metric("disc/ref_logits", mean_disc_ref_logits)
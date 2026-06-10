from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import itertools

from .params import PPOHyperparams
from moveitmoveit.src.buffers import RolloutBuffer, Transition
from moveitmoveit.src.models import PPONetworks
from utils import Logger

from .base import BaseAlgo

class PPO(BaseAlgo):
    networks: PPONetworks
    params: PPOHyperparams

    def __init__(
        self,
        networks: PPONetworks,
        params: PPOHyperparams,
        logger: Logger,
    ):
        super().__init__(networks, params, logger)

        self.transition = Transition()
        self._update_count = 0
        self._prev_terminated = None

    def init_storage(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.storage = RolloutBuffer(
            num_envs=num_envs,
            num_transitions=num_transitions,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.networks.device,
        )

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.networks.record_obs(observations.detach())
        normed_obs = self.networks.normalize_obs(observations)

        normed_actions = self.networks.act(normed_obs, deterministic)
        actions = self.networks.unnormalize_actions(normed_actions)

        self.transition.actions = normed_actions.detach()
        self.transition.values = self.networks.crit(normed_obs).detach()
        self.transition.actions_log_prob = self.networks.get_actions_log_prob(normed_actions).detach()
        self.transition.action_mean = self.networks.action_mean.detach()
        self.transition.action_sigma = self.networks.action_std.detach()
        self.transition.observations = normed_obs.detach()
        
        return actions 

    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        self.logger.log_metric("reward/total_mean", rewards.mean().item())
        self.logger.log_metric("reward/total_std", rewards.std().item())

        if self._prev_terminated is None:
            self._prev_terminated = torch.zeros_like(terminated, dtype=torch.bool)

        # With NEXT_STEP autoreset, the step after a termination is fabricated:
        # the env discarded the action and returned the new episode's reset obs.
        # Mask those transitions out of the PPO update.
        self.transition.valid = ~self._prev_terminated
        self.logger.log_metric("ppo/masked_fraction", self._prev_terminated.float().mean().item())

        self.transition.rewards = rewards.clone().detach()
        self.transition.terminated = terminated.clone().detach()
        self.transition.truncated = truncated.clone().detach()
        self.storage.add(self.transition)
        self.transition = Transition()

        self._prev_terminated = terminated.clone().detach().bool()

    def compute_returns(self, last_values: torch.Tensor) -> None:
        advantage = 0

        # Compute advantages backwards through time using GAE
        for step in reversed(range(self.storage.size)):
            if step == self.storage.size - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[step + 1]

            next_is_not_terminal = 1.0 - self.storage.terminated[step].float()

            # Temporal difference error: δ = r + γV(s') - V(s)
            delta = (
                self.storage.rewards[step]
                + next_is_not_terminal * self.params.discount * next_values
                - self.storage.values[step]
            )

            # GAE advantage: A = δ + γλA_{t+1}
            advantage = delta + next_is_not_terminal * self.params.discount * self.params.td_lambda * advantage

            # Return: R = A + V
            self.storage.returns[step] = advantage + self.storage.values[step]

        advantage = self.storage.returns - self.storage.values
        self.storage.add_advantage(advantage)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        normed_obs = self.networks.normalize_obs(observations)
        return self.networks.crit(normed_obs)

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        super().update(optimizer)

        # Accumulators for mean metrics
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_approx_kl = 0
        mean_exact_kl = 0
        mean_clip_fraction = 0
        mean_explained_variance = 0
        mean_grad_norm_before_clip = 0
        mean_grad_norm_after_clip = 0

        # Ratio statistics
        mean_ratio_mean = 0
        mean_ratio_std = 0
        mean_ratio_min = 0
        mean_ratio_max = 0

        # Log probability statistics
        mean_old_log_prob = 0
        mean_new_log_prob = 0
        mean_old_log_prob_std = 0
        mean_new_log_prob_std = 0

        # Advantage statistics (pre-normalization)
        mean_advantage_mean = 0
        mean_advantage_std = 0
        mean_advantage_min = 0
        mean_advantage_max = 0

        # Effective sample size
        mean_ess = 0

        # Per-dimension accumulators (initialized on first batch)
        action_dim = None
        mean_mu_mean_per_dim = None
        mean_mu_std_per_dim = None
        mean_sigma_mean_per_dim = None
        mean_sigma_std_per_dim = None
        mean_entropy_per_dim = None

        generator = self.storage.mini_batch_generator(
            num_mini_batches=self.params.num_mini_batches,
            num_epochs=self.params.num_learning_epochs,
        )

        for (
            observations_batch,
            actions_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            old_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:

            # --- Advantage statistics (pre-normalization) ---
            mean_advantage_mean += advantages_batch.mean().item()
            mean_advantage_std += advantages_batch.std().item()
            mean_advantage_min += advantages_batch.min().item()
            mean_advantage_max += advantages_batch.max().item()

            # --- Normalize advantages ---
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                    advantages_batch.std() + 1e-8
                )
            # They clip advange for some reason. WHY do they do this?
            # TODO

            # --- Forward passes ---
            self.networks.act(observations_batch) #update distribution from OLD samples
            actions_log_prob_batch = self.networks.get_actions_log_prob(actions_batch) #get action log probability
            value_batch = self.networks.crit(observations_batch)

            mu_batch = self.networks.action_mean
            sigma_batch = self.networks.action_std
            entropy_batch = self.networks.actor.entropy

            # --- Importance sampling ratio ---
            ratio = torch.exp(actions_log_prob_batch - old_log_prob_batch)

            # --- Ratio statistics ---
            mean_ratio_mean += ratio.mean().item()
            mean_ratio_std += ratio.std().item()
            mean_ratio_min += ratio.min().item()
            mean_ratio_max += ratio.max().item()

            # --- Log probability statistics ---
            mean_old_log_prob += old_log_prob_batch.mean().item()
            mean_new_log_prob += actions_log_prob_batch.mean().item()
            mean_old_log_prob_std += old_log_prob_batch.std().item()
            mean_new_log_prob_std += actions_log_prob_batch.std().item()

            # --- Approximate KL (first-order Taylor expansion) ---
            with torch.no_grad():
                approx_kl = (old_log_prob_batch - actions_log_prob_batch).mean()
                mean_approx_kl += approx_kl.item()

            # --- Exact KL between two diagonal Gaussians ---
            # KL(π_old || π_new) = Σ_d [ log(σ_new/σ_old) + (σ_old² + (μ_old - μ_new)²) / (2σ_new²) - 1/2 ]                                                                                                                                                                                                
            with torch.no_grad():
                exact_kl = torch.sum(
                    torch.log(sigma_batch / (old_sigma_batch + 1e-5) + 1e-5)
                    + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
                    / (2.0 * sigma_batch.pow(2))
                    - 0.5,
                    dim=-1,
                ).mean()
                mean_exact_kl += exact_kl.item()

            # --- Clip fraction ---
            with torch.no_grad():
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.params.clip_param).float().mean()
                )
                mean_clip_fraction += clip_fraction.item()

            # --- Effective sample size (normalized to [0, 1]) ---
            with torch.no_grad():
                weights = ratio / (ratio.sum() + 1e-8)
                ess = 1.0 / (weights.pow(2).sum() * len(weights) + 1e-8)
                mean_ess += ess.item()

            # --- Explained variance ---
            with torch.no_grad():
                y_pred = value_batch.flatten()
                y_true = returns_batch.flatten()
                explained_var = 1.0 - (y_true - y_pred).var() / (y_true.var() + 1e-8)
                mean_explained_variance += explained_var.item()

            # --- Per-dimension policy statistics ---
            with torch.no_grad():
                if action_dim is None:
                    action_dim = mu_batch.shape[-1]
                    mean_mu_mean_per_dim = torch.zeros(action_dim)
                    mean_mu_std_per_dim = torch.zeros(action_dim)
                    mean_sigma_mean_per_dim = torch.zeros(action_dim)
                    mean_sigma_std_per_dim = torch.zeros(action_dim)
                    mean_entropy_per_dim = torch.zeros(action_dim)

                mean_mu_mean_per_dim += mu_batch.mean(dim=0).cpu()
                mean_mu_std_per_dim += mu_batch.std(dim=0).cpu()
                mean_sigma_mean_per_dim += sigma_batch.mean(dim=0).cpu()
                mean_sigma_std_per_dim += sigma_batch.std(dim=0).cpu()

                # Gaussian entropy per dimension: 0.5 * log(2πeσ²) = 0.5 * (1 + log(2π) + 2*log(σ))
                per_dim_entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + 2.0 * torch.log(sigma_batch + 1e-8))
                mean_entropy_per_dim += per_dim_entropy.mean(dim=0).cpu()

            # --- Surrogate loss (PPO clipped objective) ---
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.params.clip_param, 1.0 + self.params.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # --- Total loss ---
            actor_loss = surrogate_loss - self.params.entropy_coef * entropy_batch.mean()
            value_loss = (returns_batch - value_batch).pow(2).mean()
            total_loss = actor_loss + self.params.value_loss_coef * value_loss

            # --- Backward pass and gradient clipping ---
            optimizer.zero_grad()
            total_loss.backward()

            all_params = list(self.networks.parameters())

            grad_norm_before = torch.norm(
                torch.stack(
                    [p.grad.detach().norm() for p in all_params if p.grad is not None]
                )
            ).item()
            mean_grad_norm_before_clip += grad_norm_before

            grad_norm_after = nn.utils.clip_grad_norm_(
                all_params, self.params.max_grad_norm
            ).item()
            mean_grad_norm_after_clip += min(grad_norm_after, self.params.max_grad_norm)

            optimizer.step()

            # --- Accumulate core losses ---
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        # --- Compute means ---
        num_updates = self.params.num_learning_epochs * self.params.num_mini_batches

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_approx_kl /= num_updates
        mean_exact_kl /= num_updates
        mean_clip_fraction /= num_updates
        mean_explained_variance /= num_updates
        mean_grad_norm_before_clip /= num_updates
        mean_grad_norm_after_clip /= num_updates
        mean_ratio_mean /= num_updates
        mean_ratio_std /= num_updates
        mean_ratio_min /= num_updates
        mean_ratio_max /= num_updates
        mean_old_log_prob /= num_updates
        mean_new_log_prob /= num_updates
        mean_old_log_prob_std /= num_updates
        mean_new_log_prob_std /= num_updates
        mean_advantage_mean /= num_updates
        mean_advantage_std /= num_updates
        mean_advantage_min /= num_updates
        mean_advantage_max /= num_updates
        mean_ess /= num_updates

        if action_dim is not None:
            mean_mu_mean_per_dim /= num_updates
            mean_mu_std_per_dim /= num_updates
            mean_sigma_mean_per_dim /= num_updates
            mean_sigma_std_per_dim /= num_updates
            mean_entropy_per_dim /= num_updates

        self.storage.clear()

        # === Logging ===
        # Core losses 
        self.logger.log_metric("ppo/loss/surrogate", mean_surrogate_loss)
        self.logger.log_metric("ppo/loss/value", mean_value_loss)
        self.logger.log_metric("ppo/loss/entropy", mean_entropy)

        # Policy constraint diagnostics  
        # self.logger.log_metric("policy/approx_kl", mean_approx_kl)
        # self.logger.log_metric("policy/exact_kl", mean_exact_kl)
        # self.logger.log_metric("policy/clip_fraction", mean_clip_fraction)

        # Importance sampling diagnostics
        # self.logger.log_metric("ratio/mean", mean_ratio_mean)
        # self.logger.log_metric("ratio/std", mean_ratio_std)
        # self.logger.log_metric("ratio/min", mean_ratio_min)
        # self.logger.log_metric("ratio/max", mean_ratio_max)
        # self.logger.log_metric("ratio/effective_sample_size", mean_ess)

        # Log probability diagnostics
        # self.logger.log_metric("log_prob/old_mean", mean_old_log_prob)
        # self.logger.log_metric("log_prob/new_mean", mean_new_log_prob)
        # self.logger.log_metric("log_prob/old_std", mean_old_log_prob_std)
        # self.logger.log_metric("log_prob/new_std", mean_new_log_prob_std)

        # Advantage diagnostics (pre-normalization)
        self.logger.log_metric("ppo/advantage/mean", mean_advantage_mean)
        self.logger.log_metric("ppo/advantage/std", mean_advantage_std)
        self.logger.log_metric("ppo/advantage/min", mean_advantage_min)
        self.logger.log_metric("ppo/advantage/max", mean_advantage_max)

        # Value function quality
        # self.logger.log_metric("value/explained_variance", mean_explained_variance)

        # Gradient diagnostics
        # self.logger.log_metric("grad/norm_before_clip", mean_grad_norm_before_clip)
        # self.logger.log_metric("grad/norm_after_clip", mean_grad_norm_after_clip)

        # Per-dimension policy statistics
        # if action_dim is not None:
            # for d in range(action_dim):
                # self.logger.log_metric(f"policy_dim/mu_mean_dim{d}", mean_mu_mean_per_dim[d].item())
                # self.logger.log_metric(f"policy_dim/mu_std_dim{d}", mean_mu_std_per_dim[d].item())
                # self.logger.log_metric(f"policy_dim/sigma_mean_dim{d}", mean_sigma_mean_per_dim[d].item())
                # self.logger.log_metric(f"policy_dim/sigma_std_dim{d}", mean_sigma_std_per_dim[d].item())
                # self.logger.log_metric(f"policy_dim/entropy_dim{d}", mean_entropy_per_dim[d].item())

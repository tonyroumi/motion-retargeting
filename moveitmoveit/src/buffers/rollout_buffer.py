from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import torch

from .base_buffer import BaseBuffer


@dataclass
class Transition:
    """Container for a single environment transition."""

    observations: torch.Tensor = None
    actions: torch.Tensor = None
    rewards: torch.Tensor = None
    terminated: torch.Tensor = None
    truncated: torch.Tensor = None
    values: torch.Tensor = None
    actions_log_prob: torch.Tensor = None
    action_mean: torch.Tensor = None
    action_sigma: torch.Tensor = None
    valid: torch.Tensor = None

class RolloutBuffer(BaseBuffer):
    """Fixed-length storage for rollout data.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__(capacity=num_transitions, device=device)
        self.num_envs = num_envs
        self.num_transitions = num_transitions

        # Storage tensors – shape [T, E, D]
        self.observations = torch.zeros(num_transitions, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_transitions, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_transitions, num_envs, 1, device=device)
        self.terminated = torch.zeros(num_transitions, num_envs, 1, device=device).byte()
        self.truncated = torch.zeros(num_transitions, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions, num_envs, action_dim, device=device)
        self.sigma = torch.zeros(num_transitions, num_envs, action_dim, device=device)

        # Transitions fabricated by autoreset are marked invalid and excluded
        # from the mini-batches.
        self.valid = torch.ones(num_transitions, num_envs, 1, device=device, dtype=torch.bool)

        # Computed during advantage estimation
        self.returns = torch.zeros(num_transitions, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions, num_envs, 1, device=device)

    def add(self, transition: Transition) -> None:
        """Append one timestep across all environments."""
        if self.step >= self.num_transitions:
            raise OverflowError(
                "Rollout buffer overflow! Call clear() before adding new transitions."
            )

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.terminated[self.step].copy_(transition.terminated.view(-1, 1))
        self.truncated[self.step].copy_(transition.truncated.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if transition.valid is not None:
            self.valid[self.step].copy_(transition.valid.view(-1, 1))
        else:
            self.valid[self.step].fill_(True)
        self.step += 1

    def add_advantage(self, advantage: torch.Tensor) -> None:
        self.advantages = advantage

    def clear(self) -> None:
        self.step = 0

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor,
        ],
        None,
        None,
    ]:
        """Yield mini-batches for PPO-style optimisation.

        Flattens the ``[T, E]`` leading dimensions and randomly shuffles.

        Yields
        ------
        obs, actions, values, advantages, returns, old_log_prob, old_mu, old_sigma
        """
        # Flatten time × envs
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # Sample only valid transitions (autoreset steps are masked out).
        valid_indices = self.valid.flatten(0, 1).squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
        num_valid = valid_indices.numel()
        mini_batch_size = num_valid // num_mini_batches

        perm = torch.randperm(num_valid, device=self.device)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                idx = valid_indices[perm[start:end]]

                yield (
                    observations[idx],
                    actions[idx],
                    values[idx],
                    advantages[idx],
                    returns[idx],
                    old_log_prob[idx],
                    old_mu[idx],
                    old_sigma[idx],
                )
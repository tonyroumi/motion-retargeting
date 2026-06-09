from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from moveitmoveit.src.models.networks.actor import GaussianActor
from moveitmoveit.src.models.networks.base import BaseMLP
from moveitmoveit.src.models.norm import EmpiricalNorm
from moveitmoveit.src.models.container import NetworkContainer

@dataclass
class PPONetworks(NetworkContainer):
    actor: GaussianActor
    critic: BaseMLP
    obs_norm: EmpiricalNorm
    action_norm: EmpiricalNorm

    def to_device(self, device: str) -> None:
        self.actor.to(device)
        self.critic.to(device)
        self.obs_norm.to(device)
        self.action_norm.to(device)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor(obs, deterministic)

    def crit(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def normalize_obs(self, obs: torch.Tensor) -> None:
        return self.obs_norm.normalize(obs)

    def unnormalize_actions(self, actions: torch.Tensor) -> None:
        return self.action_norm.unnormalize(actions)

    def record_obs(self, obs: torch.Tensor) -> None:
        self.obs_norm.record(obs)

    def update_normalizers(self) -> None:
        self.obs_norm.update()

    @property
    def action_mean(self) -> torch.Tensor:
        return self.actor.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.actor.action_std

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.actor.get_actions_log_prob(actions)

    def parameters(self) -> Iterator[nn.Parameter]:
        return itertools.chain(self.actor.parameters(), self.critic.parameters())

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return itertools.chain(self.actor.trainable_parameters(), self.critic.trainable_parameters())

    def state_dict(self) -> dict:
        d = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        d["obs_norm"] = self.obs_norm.state_dict()
        d["action_norm"] = self.action_norm.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.obs_norm.load_state_dict(state["obs_norm"])
        self.action_norm.load_state_dict(state["action_norm"])

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.obs_norm.eval()
        self.action_norm.eval()

    def train(self) -> None:
        self.actor.train()
        self.critic.train()
        self.obs_norm.train()
        self.action_norm.train()
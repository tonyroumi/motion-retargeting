from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn

from moveitmoveit.src.models.networks.base import BaseMLP
from moveitmoveit.src.models.norm import EmpiricalNorm
from moveitmoveit.src.models.ppo import PPONetworks

@dataclass
class AMPNetworks(PPONetworks):
    discriminator: BaseMLP
    disc_obs_norm: EmpiricalNorm

    def to_device(self, device: str) -> None:
        super().to_device(device)
        self.discriminator.to(device)
        self.disc_obs_norm.to(device)

    def disc(self, disc_obs: torch.Tensor) -> torch.Tensor:
        return self.discriminator(disc_obs)

    def normalize_disc_obs(self, disc_obs: torch.Tensor) -> None:
        return self.disc_obs_norm.normalize(disc_obs)

    def record_disc_obs(self, disc_obs: torch.Tensor) -> None:
        self.disc_obs_norm.record(disc_obs)

    def update_normalizers(self) -> None:
        super().update_normalizers()
        self.disc_obs_norm.update()
    
    def get_disc_logit_weights(self) -> torch.Tensor:
        return self.discriminator.get_logit_weights()

    def parameters(self) -> Iterator[nn.Parameter]:
        return super().parameters()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return itertools.chain(super().trainable_parameters(), self.discriminator.trainable_parameters())

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["discriminator"] = self.discriminator.state_dict()
        d["disc_obs_norm"] = self.disc_obs_norm.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.discriminator.load_state_dict(state["discriminator"])
        self.disc_obs_norm.load_state_dict(state["disc_obs_norm"])

    def eval(self) -> None:
        super().eval()
        self.discriminator.eval()
        self.disc_obs_norm.eval()

    def train(self) -> None:
        super().train()
        self.discriminator.train()
        self.disc_obs_norm.train()
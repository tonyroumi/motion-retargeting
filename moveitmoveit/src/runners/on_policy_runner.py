from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List

from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from utils import Logger
from moveitmoveit.src.algo.base import BaseAlgo
from moveitmoveit.src.types import BaseParams

@dataclass(frozen=True)
class OnPolicyRunnerParams(BaseParams):
    total_timesteps: int = 2_000_000
    num_transitions_per_env: int = 32

    log_interval: int = 1           # iterations between console logs
    checkpoint_interval: int = 100  # iterations between saves

    device: str = "cpu"

class OnPolicyRunner:
    """Generic on-policy training loop. """

    def __init__(
        self,
        environment: AsyncVectorEnv,
        algorithm: BaseAlgo,
        params: OnPolicyRunnerParams,
        logger: Logger,
    ):
        self.env = environment
        self.algo = algorithm
        self.params = params
        self.logger = logger

        obs_dim = environment.observation_space.shape[-1]
        action_dim = environment.action_space.shape[-1]

        self.algo.to_device(self.params.device)
        self.algo.init_storage(
            num_envs=self.env.num_envs,
            num_transitions=params.num_transitions_per_env,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.optimizer = optim.Adam(
            self.algo.networks.trainable_parameters(),
            lr=algorithm.params.lr,
        )

        self.current_timestep = 0
        self.current_iteration = 0

    def learn(self) -> None:
        self.algo.train()
        
        steps_per_iter = self.env.num_envs * self.params.num_transitions_per_env
        total_iterations = self.params.total_timesteps // steps_per_iter

        obs, info = self.env.reset()
        self.algo.process_reset(info)

        train_start = time.perf_counter()

        for iteration in range(total_iterations):
            # collect rollouts
            for _ in range(self.params.num_transitions_per_env):
                with torch.no_grad():
                    actions = self.algo.act(obs)

                obs, reward, terminated, truncated, info = self.env.step(
                    actions
                )

                # gymnasium envs auto reset
                self.algo.process_env_step(
                    rewards=reward,
                    terminated=terminated,
                    truncated=truncated,
                    infos=info,
                )

                self.logger.step()
            
            # compute returns and update
            with torch.no_grad():
                last_values = self.algo.get_value(obs)
            self.algo.compute_returns(last_values)

            self.algo.update(self.optimizer)

            self.current_timestep += steps_per_iter
            self.current_iteration += 1

            if self.current_iteration % self.params.log_interval == 0:
                self.logger.pprint(
                    iteration=self.current_iteration,
                    wall_time=time.perf_counter() - train_start,
                    samples=self.current_timestep,
                )

            if self.current_iteration % self.params.checkpoint_interval == 0:
                path = os.path.join(
                    self.logger.log_dir,
                    f"checkpoint_{self.current_iteration}.pt",
                )
                self.save(path)
                self.logger.info(f"  Checkpoint saved → {path}")

    def save(self, path: str) -> None:
        data = {
            "networks": self.algo.networks.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.current_iteration,
            "timestep": self.current_timestep,
        }
        torch.save(data, path)
        torch.save(data, os.path.join(os.path.dirname(path), "latest.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.params.device)
        self.algo.networks.load_state_dict(ckpt["networks"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_iteration = ckpt["iteration"]
        self.current_timestep = ckpt["timestep"]
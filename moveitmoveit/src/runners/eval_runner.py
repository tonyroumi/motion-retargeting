from __future__ import annotations

import time
from dataclasses import dataclass
from tracemalloc import start

import numpy as np
import torch

from gymnasium.vector import AsyncVectorEnv

from moveitmoveit.src.models import NetworkContainer
import time

from utils import Logger

FPS = 30
DT = 1.0 / FPS

@dataclass(frozen=True)
class EvalRunnerParams():
    num_eval_episodes: int = 100
    num_episode_steps: int = 600
    render: bool = False

class EvalRunner:
    """Policy evaluation runner.

    This runner:
    - loads trained network weights
    - runs policy rollouts
    - does not store transitions
    - does not compute returns
    - does not update networks
    """

    def __init__(
        self,
        environment: AsyncVectorEnv,
        network: NetworkContainer,
        params: EvalRunnerParams,
        logger: Logger,
    ):
        self.env = environment
        self.network = network
        self.params = params
        self.logger = logger

    def evaluate(self) -> dict:

        completed_returns = []
        completed_lengths = []
        eval_start = time.perf_counter()

        with torch.no_grad():
            for _ in range(self.params.num_eval_episodes):
                obs, info = self.env.reset()

                episode_returns = 0
                episode_lengths = 0

                for _ in range(self.params.num_episode_steps):
                    start = time.time()
                    normed_obs = self.network.normalize_obs(obs)
                    normed_actions = self.network.act(normed_obs, deterministic=True)
                    actions = self.network.unnormalize_actions(normed_actions)

                    obs, rewards, terminated, truncated, info = self.env.step(actions)

                    episode_returns += rewards
                    episode_lengths += 1

                    if self.params.render:
                        self.env.render()

                    elapsed = time.time() - start
                    sleep_time = max(0, DT - elapsed)
                    time.sleep(sleep_time)

                    if np.any(terminated):
                        break

                completed_returns.append(float(episode_returns))
                completed_lengths.append(int(episode_lengths))
        
        self.env.close()

        metrics = {
            "num_episodes": len(completed_returns),
            "mean_return": float(np.mean(completed_returns)),
            "std_return": float(np.std(completed_returns)),
            "min_return": float(np.min(completed_returns)),
            "max_return": float(np.max(completed_returns)),
            "mean_episode_length": float(np.mean(completed_lengths)),
        }

        for key, value in metrics.items():
            if key != "num_episodes":
                self.logger.log_metric(f"eval/{key}", value)

        self.logger.pprint(
            title=f"Evaluation ({metrics['num_episodes']} episodes)",
            wall_time=time.perf_counter() - eval_start,
            samples=metrics["num_episodes"],
        )

        return metrics
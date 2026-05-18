from functools import partial
from typing import Any, Dict, List, Callable, Tuple

import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import NumpyToTorch as NumpyToTorchSingle
from gymnasium.wrappers.vector import NumpyToTorch

import gymnasium as gym

from moveitmoveit.src.algo import PPO, PPOHyperparams, AMP, AMPHyperparams
from moveitmoveit.src.env import AMPEnv
from moveitmoveit.src.models import GaussianActor, BaseMLP, AMPNetworks, PPONetworks, NetworkContainer, EmpiricalNorm
from moveitmoveit.src.runners import OnPolicyRunner, OnPolicyRunnerParams
from utils import Logger

def make_env(cfg):
    def env_factory(
        env_id: str,
        env_idx: int,
        record_video_path: str | None = None,
        record_video_interval: int = 2000,
        env_kwargs: Dict[str, Any] | None = None,
        wrappers: List[Callable] | None = None,
    ):
        env_kwargs = env_kwargs or {}
        wrappers = wrappers or []

        def _init():
            env = gym.make(env_id, **env_kwargs)

            for wrapper in wrappers:
                env = wrapper(env)

            if record_video_path is not None and env_idx == 0:
                env = RecordVideo(
                    env,
                    record_video_path,
                    episode_trigger=lambda x: x % record_video_interval == 0,
                )

            return env

        return _init

    env_kwargs = dict(cfg.get("kwargs", {}))
    wrappers = list(cfg.get("wrappers", []))
    num_envs = cfg.get("num_envs", 1)
    seed = cfg.get("seed", 0)

    env_type = cfg.type.lower()

    if env_type == "jax":
        raise NotImplementedError("Jax environment has not been implemented yet")
    elif env_type == "gym:cpu":
        vector_env_cls = partial(AsyncVectorEnv, context="fork")
        env = vector_env_cls([
            env_factory(
                env_id=cfg.id,
                env_idx=idx,
                env_kwargs=env_kwargs,
                wrappers=wrappers,
            )
            for idx in range(num_envs)
        ])
        env = NumpyToTorch(env, device="cuda")
    elif env_type == "gym:play":
        env = gym.make(cfg.id, render_mode="human", **env_kwargs)
        env = NumpyToTorchSingle(env, device="cuda")
    else:
        raise ValueError(f"Unknown env type '{env_type}'. Available: ['gym:cpu', 'jax']")

    env.reset(seed=seed)
    return env

def make_networks(
    cfg,
    in_channels: int,
    out_channels: int,
    action_space: Tuple,
) -> NetworkContainer:
    """Factory that builds a NetworkContainer with normalizers."""
    actor = GaussianActor(in_channels=in_channels, out_channels=out_channels, **cfg.actor)
    critic = BaseMLP(in_channels=in_channels, out_channels=1, **cfg.critic)

    obs_norm = None
    if cfg.get("obs_norm") is not None:
        obs_norm = EmpiricalNorm(shape=(in_channels,), **cfg.obs_norm)

    action_norm = None
    if cfg.get("action_norm") is not None:
        a_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype=torch.float32)
        a_std = torch.tensor(0.5 * (action_space.high - action_space.low), dtype=torch.float32)
        action_norm = EmpiricalNorm(
            shape=a_mean.shape,
            init_mean=a_mean,
            init_std=a_std,
            **cfg.action_norm,
        )

    if cfg.get("discriminator") is not None:
        discriminator = BaseMLP(in_channels=10 * in_channels, out_channels=1, **cfg.discriminator)

        disc_obs_norm = None
        if cfg.get("disc_obs_norm") is not None:
            disc_obs_norm = EmpiricalNorm(shape=(10*in_channels,), **cfg.disc_obs_norm)

        return AMPNetworks(
            actor=actor,
            critic=critic,
            discriminator=discriminator,
            obs_norm=obs_norm,
            action_norm=action_norm,
            disc_obs_norm=disc_obs_norm,
        )

    return PPONetworks(actor=actor, critic=critic, obs_norm=obs_norm, action_norm=action_norm)

def make_algo(cfg, networks: NetworkContainer, logger: Logger):
    name = cfg.name.lower()

    if name == "ppo":
        algo_cls, params_cls = PPO, PPOHyperparams
    if name == "amp":
        algo_cls, params_cls = AMP, AMPHyperparams

    params = params_cls.from_dict(cfg)
    return algo_cls(networks=networks, params=params, logger=logger)

def make_runner(cfg, env, algo, logger: Logger):
    params = OnPolicyRunnerParams.from_dict(cfg)
    return OnPolicyRunner(environment=env, algorithm=algo, params=params, logger=logger)
from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from moveitmoveit.src.factory import make_env, make_algo, make_networks, make_runner
from utils import Logger

@hydra.main(version_base=None, config_path="./configs/amp", config_name="humanoid")
def _train(cfg: DictConfig) -> None:
    logger = Logger(HydraConfig.get().runtime.output_dir)

    env = make_env(cfg.environment)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    networks = make_networks(
        cfg.networks,
        in_channels=obs_dim,
        out_channels=action_dim,
        action_space=env.single_action_space,
    )
    algo = make_algo(cfg.algorithm, networks, logger)
    runner = make_runner(cfg.runner, env, algo, logger)

    runner.learn()

def _find_checkpoint(checkpoint_dir: str) -> str:
    candidates = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir!r}")
    return os.path.join(checkpoint_dir, candidates[-1])

def _play(checkpoint_dir: str, num_episodes: int, render: bool) -> None:
    from moveitmoveit.src.runners import EvalRunner, EvalRunnerParams

    config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in {checkpoint_dir!r}. "
            "Ensure the checkpoint was saved by this training script."
        )

    cfg = OmegaConf.load(config_path)

    logger = Logger(log_dir=checkpoint_dir)

    play_env_cfg = OmegaConf.merge(cfg.environment, {"num_envs": 1, "type": "gym:play"})
    env = make_env(play_env_cfg)

    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    networks = make_networks(
        cfg.networks,
        in_channels=obs_dim,
        out_channels=action_dim,
        action_space=env.action_space,
    )
    networks.to_device(cfg.runner.device)

    ckpt_path = _find_checkpoint(checkpoint_dir)
    ckpt = torch.load(ckpt_path, map_location=cfg.runner.device, weights_only=False)
    networks.load_state_dict(ckpt["networks"])
    logger.info(f"Loaded {ckpt_path}  (iter={ckpt['iteration']}, steps={ckpt['timestep']:,})")

    eval_params = EvalRunnerParams(
        num_eval_episodes=num_episodes,
        render=render,
    )
    eval_runner = EvalRunner(
        environment=env,
        network=networks,
        params=eval_params,
        logger=logger,
    )
    eval_runner.evaluate()
    logger.close()

def _build_play_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py play",
        description="Play a trained policy from a checkpoint directory.",
    )
    p.add_argument("mode")
    p.add_argument(
        "--checkpoint", required=True, metavar="DIR",
        help="Directory produced by training (contains config.yaml + *.pt files)",
    )
    p.add_argument(
        "--episodes", type=int, default=10, metavar="N",
        help="Number of episodes to evaluate (default: 10)",
    )
    p.add_argument(
        "--render", action="store_true",
        help="Render the environment during evaluation",
    )
    return p


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        args = _build_play_parser().parse_args()
        _play(args.checkpoint, args.episodes, args.render)
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "train":
            sys.argv.pop(1)
        _train()

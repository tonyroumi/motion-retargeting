from .mujoco_env import MujocoEnv, MujocoEnvParams
from .amp_env import AMPEnv

from gymnasium.envs.registration import register

register(
    id="AMPEnv",
    entry_point=AMPEnv,
)

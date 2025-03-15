from dataclasses import dataclass
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import numpy as np
from cocogrid.utils.logging import LoggingWrapper, CocogridLogger

from gcrs.utils.recording import RecordVideo
from gcrs.utils.logging import SubgoalLogger


@dataclass
class CocoGridEnv:
    agent: str = "box2d"
    """The type of the walker, from 'box2d', 'mujoco/ball', 'mujoco/square', 'mujoco/ant'"""
    observation: str = "no-arena"
    """Specifies the observation type, from 'full', 'no-arena', 'object-one-hot'"""
    xy_scale: float = 1
    """The arena scale (minimum based on walker type)"""
    seed: int = None
    """The seed for the arena generation"""
    render_width: int = 256
    """How big should the render be?"""


def make_cocogrid_env(
    env_id,
    idx,
    capture_video,
    run_name,
    gamma,
    env_kwargs={},
    logging_params=None,
    video_dir=None,
    goal_version="dense-v1",
    reward_scale=1,
    norm_obs=False,
    norm_reward=False,
    is_eval=False,
):
    from gcrs.goal import wrap_env_with_goal

    if video_dir is None:
        video_dir = f"videos/{run_name}"
    video_prefix = "eval" if is_eval else "train"
    video_trigger = (
        (
            lambda ep: (
                (ep in [0, 1, 2, 4, 6, 8, 12, 16, 20, 24]) or ep > 24 and (ep % 8 == 0)
            )
        )
        if is_eval
        else None
    )

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = wrap_env_with_goal(env, env_id, goal_version, gamma)
            env = RecordVideo(
                env, video_dir, name_prefix=video_prefix, episode_trigger=video_trigger
            )
        else:
            env = gym.make(env_id, **env_kwargs)
            env = wrap_env_with_goal(env, env_id, goal_version, gamma)

        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        if norm_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10)
            )

        if norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        elif reward_scale != 1:
            env = gym.wrappers.TransformReward(env, lambda r: r * reward_scale)

        if logging_params is not None:
            env = LoggingWrapper(
                env,
                logging_params["writer"],
                max_timesteps=logging_params["max_steps"],
                standard_label=logging_params["prefix"],
                is_eval=is_eval,
            )
            # for logger in get_minimujo_heatmap_loggers(env, gamma=0.99):
            #     logger.label = f'{logging_params["prefix"]}_{logger.label}'.lstrip('_')
            #     env.subscribe_metric(logger)
            prefix = logging_params["prefix"]
            env.subscribe_metric(CocogridLogger(prefix))
            env.subscribe_metric(SubgoalLogger(prefix))
        return env

    return thunk


def get_environment_constructor_for_id(env_id: str) -> Callable:
    if is_cocogrid_env(env_id):
        return make_cocogrid_env
    raise ValueError(f"No environment constructor defined for {env_id}.")


def is_cocogrid_env(env_id: str) -> bool:
    return env_id.lower().startswith("cocogrid")

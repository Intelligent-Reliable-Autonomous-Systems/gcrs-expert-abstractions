from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import minimujo

@dataclass
class MinimujoEnv:
    walker_type: str = 'ball'
    """The type of the walker, from 'ball', 'square', 'ant', 'humanoid'"""
    xy_scale: float = 1
    """The arena scale (minimum based on walker type)"""
    seed: int = None
    """The seed for the arena generation"""
    random_spawn: bool = False
    """Whether to randomly position the walker on reset"""
    random_rotation: bool = False
    """Whether to randomly rotate the walker on reset"""
    observation_type: str = "pos,vel,walker"
    """What type of observation should the environment emit? Options are 'top_camera', 'walker', 'pos', 'vel', 'goal'. Can be combined as a comma seprated list (e.g. 'pos,walker')"""
    image_observation_format: str = '0-1'
    """Whether the image should be formatted in range 0-1 or 0-255"""
    render_width: int = 256
    """How big should the render be?"""

def is_minimujo_env(env_id: str):
    return env_id.lower().startswith("minimujo")

class EpisodePadWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._observation = None
        self._terminated = False
        self._truncated = False
        self._info = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._observation = None
        self._terminated = False
        self._truncated = False
        self._info = {}

        return super().reset(seed=seed, options=options)
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            return self._observation, 0, self._terminated, self._truncated, self._info

        obs, reward, term, trunc, info = super().step(action)
        if term or trunc:
            self._observation = obs
            self._terminated = term
            self._truncated = trunc
            self._info = info

        return obs, reward, term, trunc, info
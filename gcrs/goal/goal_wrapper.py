from __future__ import annotations

from typing import Any, Callable, Dict, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from gcrs.goal.planner import SubgoalPlanner, AbstractState, Observation


class GoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        abstraction: Callable[[Observation, gym.Env], AbstractState],
        planner: SubgoalPlanner,
        observer: GoalObserver,
        use_base_reward: bool = True,
        clip_reward: int = 1000,
    ) -> None:
        super().__init__(env)

        self._abstraction = abstraction
        self._planner = planner
        self._observer = observer
        self._use_base_reward = use_base_reward
        self._clip_reward = clip_reward

        base_obs_space = self.env.unwrapped.observation_space
        self.observation_space = observer.transform_observation_space(base_obs_space)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(*args, **kwargs)

        abstract_state = self._abstraction(obs, self.env)
        self._planner.init_task(obs, abstract_state, self.env)
        self._planner.update_state(abstract_state)
        self._planner.update_plan()

        self._initial_plan_cost = self._planner.cost

        return np.concatenate([obs, self._observer.observe(self.goal)]), info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)

        prev_abstract_state = self.abstract_state
        prev_subgoal = self.subgoal
        prev_cost = self._planner.cost
        abstract_state = self._abstraction(obs, self.env)

        is_new_goal = prev_abstract_state != abstract_state
        goal_achieved = abstract_state == self.subgoal

        self._planner.update_state(abstract_state)
        self._planner.update_plan()

        info["task_reward"] = rew
        if not self._use_base_reward:
            rew = 0
        rew += self.extra_reward(
            obs, prev_abstract_state, prev_subgoal, prev_cost, term
        )
        rew = max(
            -self._clip_reward, min(rew, self._clip_reward)
        )  # ensure reward doesn't spike (e.g. if plan cost is infinity)

        info["goal"] = self.subgoal
        info["goal_achieved"] = goal_achieved
        info["is_new_goal"] = is_new_goal
        info["num_subgoals"] = self._planner.cost
        info["frac_subgoals"] = (
            self._planner.cost / self._initial_plan_cost
            if self._initial_plan_cost != 0
            else 0
        )

        return (
            np.concatenate([obs, self._observer.observe(self.subgoal)]),
            rew,
            term,
            trunc,
            info,
        )

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        terminated: bool,
    ) -> float:
        """Override this function to do reward shaping"""
        return 0

    @property
    def goal(self) -> AbstractState:
        return self._planner.goal

    @property
    def abstract_state(self) -> AbstractState:
        return self._planner.state

    @property
    def subgoal(self) -> AbstractState:
        return self._planner.next_state


class GoalObserver:
    def __init__(self, goal_observation_fn, low, high):
        self.goal_observation_fn = goal_observation_fn
        self.low = low
        self.high = high

    def transform_observation_space(self, observation_space: gym.Space):
        if (
            isinstance(observation_space, gym.spaces.Box)
            and len(observation_space.shape) == 1
        ):
            new_low = np.concatenate([observation_space.low, self.low], axis=None)
            new_high = np.concatenate([observation_space.high, self.high], axis=None)
            return gym.spaces.Box(
                low=new_low, high=new_high, dtype=observation_space.dtype
            )
        elif isinstance(observation_space, gym.spaces.Dict):
            return observation_space

    def observe(self, goal: AbstractState) -> np.ndarray:
        return self.goal_observation_fn(goal)

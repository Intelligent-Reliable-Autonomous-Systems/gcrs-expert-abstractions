from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from gcrs.goal.planner import SubgoalPlanner, AbstractState, Observation


class GoalWrapper(gym.Wrapper):
    """Wrap a gymnasium environment to augment observations with abstract subgoals and reward shaping"""

    def __init__(
        self,
        env: gym.Env,
        abstraction: Callable[[Observation, gym.Env], AbstractState],
        planner: SubgoalPlanner,
        observer: GoalObserver,
        use_base_reward: bool = True,
        clip_reward: float = 1000,
    ) -> None:
        """Construct a GoalWrapper

        Input:
        env (gym.Env): A gymnasium environment
        abstraction: A function that takes an observation and environment and returns an abstract state to be used in the planner.
        planner (SubgoalPlanner): A planner that can accept a task function and plan a sequence of abstract states.
        observer (GoalObserver): An object that transforms an abstract state into a suitable representation (e.g. a numpy vector)
        use_base_reward (bool): Whether to allow the base environment reward to pass through.
        clip_reward (float): An absolute threshold to prevent single step rewards from getting crazy.
        """
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
        # Get the abstract state
        abstract_state = self._abstraction(obs, self.env)

        # Update the planner
        self._planner.update_state(abstract_state)
        self._planner.update_plan()

        # Calculate rewards
        info["task_reward"] = rew
        if not self._use_base_reward:
            rew = 0
        rew += self.extra_reward(
            obs, prev_abstract_state, prev_subgoal, prev_cost, term
        )
        rew = max(
            -self._clip_reward, min(rew, self._clip_reward)
        )  # ensure reward doesn't spike (e.g. if plan cost is infinity)

        # Log information
        info["goal"] = self.subgoal
        info["goal_achieved"] = abstract_state == self.subgoal
        info["is_new_goal"] = prev_abstract_state != abstract_state
        info["num_subgoals"] = self._planner.cost
        info["frac_subgoals"] = (
            self._planner.cost / self._initial_plan_cost
            if self._initial_plan_cost != 0
            else 0
        )

        # Augment the observation with the subgoal
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
        """Override this function to do reward shaping.

        Inputs:
        obs: an observation from the environment this timestep.
        prev_abstract: the abstract state from the previous timestep.
        prev_subgoal: the subgoal planned in the previous timestep.
        prev_cost: the cost of the plan from the previous timestep.
        terminated: whether the environment terminated this timestep.
        """
        return 0

    @property
    def goal(self) -> AbstractState:
        """Get the final abstract goal planned by the planner."""
        return self._planner.goal

    @property
    def abstract_state(self) -> AbstractState:
        """Get the current abstract state."""
        return self._planner.state

    @property
    def subgoal(self) -> AbstractState:
        """Get the next abstract state planned by the planner."""
        return self._planner.next_state


class GoalObserver:
    """Control how observations are augmented."""

    def __init__(
        self,
        goal_observation_fn: Callable[[AbstractState], np.ndarray],
        low: Sequence[float],
        high: Sequence[float],
    ) -> None:
        """Construct a GoalObserver.

        Inputs:
        goal_observation_fn:
        """
        self.goal_observation_fn = goal_observation_fn
        self.low = low
        self.high = high

    def transform_observation_space(
        self, observation_space: gym.Space
    ) -> gym.spaces.Space:
        """Augment the base observation space with an abstract goal.
        
        Input:
        observation_space: the base environment's observation space.

        Output: an observation space of the same type, augmented to fit the goal.
        """
        if (
            isinstance(observation_space, gym.spaces.Box)
            and len(observation_space.shape) == 1
        ):
            # Observation space is single dimension. Concatenate goal
            new_low = np.concatenate([observation_space.low, self.low], axis=None)
            new_high = np.concatenate([observation_space.high, self.high], axis=None)
            return gym.spaces.Box(
                low=new_low, high=new_high, dtype=observation_space.dtype
            )
        elif isinstance(observation_space, gym.spaces.Dict):
            return observation_space
        raise NotImplementedError(
            f"Observation space {type(observation_space)} is not supported."
        )

    def observe(self, goal: AbstractState) -> np.ndarray:
        """Transform an abstract state into an array to be concatenated to the goal."""
        return self.goal_observation_fn(goal)

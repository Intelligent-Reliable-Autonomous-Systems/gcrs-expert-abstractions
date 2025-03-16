import re
from typing import Callable, Dict, Type
import minigrid.envs as envs
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import gymnasium as gym

from cocogrid.common.abstraction import GridAbstraction
from cocogrid.common.entity import get_color_id, ObjectEnum
from cocogrid.tasks import extract_feature_from_mission_text, COLOR_NAMES, OBJECT_NAMES
from cocogrid.common.multitask import MultiTaskEnv
from cocogrid.minigrid import (
    UMazeEnv,
    HallwayChoiceEnv,
    RandomCornerEnv,
    ObjectDeliveryEnv,
)
from cocogrid.minigrid.doorkeycrossing import DoorKeyCrossingEnv

from gcrs.goal.goal_wrapper import GoalWrapper, GoalObserver
from gcrs.goal.planner import AStarPlanner, AbstractState, Observation


def goal_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env):
    """Get an abstract state evaluator to signal when a goal (or lava) has been reached."""

    def eval_state(abstract: GridAbstraction):
        grid = abstract.walker_grid_cell
        if grid == GridAbstraction.GRID_GOAL:
            return 1, True
        elif grid == GridAbstraction.GRID_LAVA:
            return -1, True
        return 0, False

    return eval_state


def object_delivery_task_getter(
    obs: Observation,
    abstract: GridAbstraction,
    env: gym.Env,
    return_to_center: bool = False,
):
    """Get an abstract state evaluator to signal when the object has been delivered.
    
    Optionally, return_to_center requires the agent to return to the center square.
    """
    task = env.unwrapped.task
    pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
    matches = re.search(pattern, task)

    if not matches:
        raise Exception(f"Task '{task}' does not meet specification for ObjectDelivery")
    color = matches.group(1)
    color_idx = get_color_id(color)
    class_name = matches.group(2)
    class_idx = ObjectEnum.get_id(class_name)
    x = int(matches.group(3))
    y = int(matches.group(4))
    target_object = (class_idx, x, y, color_idx, 0)

    if return_to_center:

        def eval_state(abstract: GridAbstraction):
            if abstract.walker_pos == (2, 2):
                for obj in abstract.objects:
                    if obj == target_object:
                        return 1, True
            return 0, False
    else:

        def eval_state(abstract: GridAbstraction):
            for obj in abstract.objects:
                if obj == target_object:
                    return 1, True
            return 0, False

    return eval_state


def random_object_center_task_getter(
    obs: Observation, abstract: GridAbstraction, env: gym.Env
):
    """Curry object_delivery_task_getter with return_to_center=True."""
    return object_delivery_task_getter(obs, abstract, env, return_to_center=True)


def pickup_object_task_getter(
    obs: Observation, abstract: GridAbstraction, env: gym.Env, strict=False
):
    """Get an abstract state evaluator to signal when the task object has been picked up.
    
    Optionally, strict signals failure if the wrong object is picked up.
    """
    task = env.unwrapped.task
    color_name = extract_feature_from_mission_text(task, COLOR_NAMES)
    color_idx = get_color_id(color_name)
    type_name = extract_feature_from_mission_text(task, OBJECT_NAMES)
    type_idx = ObjectEnum.get_id(type_name)

    def eval_state(abstract: GridAbstraction):
        for obj_type, _, _, color, state in abstract.objects:
            # if is a held object
            if obj_type != GridAbstraction.DOOR_IDX and state == 1:
                # if matches target
                if color == color_idx and obj_type == type_idx:
                    return 1, True
                elif strict:
                    # in strict mode, only allow picking up target
                    return -1, True
        return 0, False

    return eval_state


def pickup_object_strict_task_getter(
    obs: Observation, abstract: GridAbstraction, env: gym.Env
):
    """Curry pickup_object_task_getter with strict=True."""
    return pickup_object_task_getter(obs, abstract, env, strict=True)


def infer_task_goal(obs, abstract, _env):
    """For multitask environments, infer the type of task from the minigrid environment."""
    goal_fn = COCOGRID_ABSTRACT_GOALS[type(_env.unwrapped.minigrid.unwrapped)]
    return goal_fn(obs, abstract, _env)

# Register default tasks for environments.
COCOGRID_ABSTRACT_GOALS: Dict[MiniGridEnv, Callable] = {
    envs.FetchEnv: pickup_object_strict_task_getter,
    envs.KeyCorridorEnv: pickup_object_task_getter,
    envs.UnlockPickupEnv: pickup_object_task_getter,
    envs.BlockedUnlockPickupEnv: pickup_object_task_getter,
    envs.CrossingEnv: goal_task_getter,
    envs.DoorKeyEnv: goal_task_getter,
    envs.FourRoomsEnv: goal_task_getter,
    envs.EmptyEnv: goal_task_getter,
    envs.DistShiftEnv: goal_task_getter,
    envs.LavaGapEnv: goal_task_getter,
    envs.LockedRoomEnv: goal_task_getter,
    envs.MultiRoomEnv: goal_task_getter,
    MultiTaskEnv: infer_task_goal,
    UMazeEnv: goal_task_getter,
    HallwayChoiceEnv: goal_task_getter,
    ObjectDeliveryEnv: random_object_center_task_getter,
    RandomCornerEnv: goal_task_getter,
    DoorKeyCrossingEnv: goal_task_getter,
}


def construct_grid_goal_wrapper(
    env: gym.Env,
    env_id: str,
    goal_cls: Type[GoalWrapper] = GoalWrapper,
    observe_subgoal=True,
):
    """Construct a GridGoalWrapper based on the grid abstraction

    Input:
    env (gym.Env): A CocoGrid gymnasium environment.
    env_id (str): The environment id used to create the environmnet.
    goal_cls (Type[GoalWrapper]): A subclass of GoalWrapper to manage abstractions, rewards, and observations.
    observe_subgoal (bool): Can be set to false to make the agent unaware of the plan (e.g. as a baseline)
    """

    # Abstraction function to map continuous to discrete.
    def abstraction_fn(obs, _env):
        # Use the state rather than observation to be consistent across observation types.
        state = _env.unwrapped.state
        return GridAbstraction.from_cocogrid_state(state)

    minigrid_env = env.unwrapped.minigrid
    # Goal observation function to map the abstracted state into a observation representation (e.g. a vector).
    if observe_subgoal:

        def goal_obs_fn(abstract: GridAbstraction):
            held_type, held_color = -1, -1
            if abstract.held_object_idx >= 0:
                held = abstract.objects[abstract.held_object_idx]
                held_type, held_color = held[0], held[3]
            return (*abstract.walker_pos, held_type, held_color)

        low = [0, 0, 0, 0]
        high = [
            minigrid_env.grid.width,
            minigrid_env.grid.height,
            len(OBJECT_NAMES),
            len(COLOR_NAMES),
        ]
        observer = GoalObserver(goal_obs_fn, low, high)
    else:
        observer = GoalObserver(lambda abstract: [], [], [])

    # Get the task-specific goal function for this environment.
    goal_fn = COCOGRID_ABSTRACT_GOALS[type(minigrid_env)]

    # Define the planner that plans over a network of abstract states, give a particular task.
    def task_edge_getter(obs: Observation, abstract: AbstractState, env: gym.Env):
        """For a given task, get the neighbors of an abstract state and evaluate each for success/failure."""
        eval_state = goal_fn(obs, abstract, env)

        def get_edges(abstract: AbstractState):
            for neighbor in abstract.get_neighbors():
                reward, term = eval_state(neighbor)
                if term:
                    if reward > 0:
                        # reached goal
                        yield neighbor, 1, True
                        continue
                    elif reward < 0:
                        # failure state
                        yield neighbor, np.inf, False
                        continue
                yield neighbor, 1, False

        return get_edges

    planner = AStarPlanner(task_edge_getter)

    return goal_cls(env, abstraction_fn, planner, observer)

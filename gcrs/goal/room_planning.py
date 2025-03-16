from functools import partial
import re
from typing import Callable, Dict, Type

from cocogrid.minigrid import (
    UMazeEnv,
    HallwayChoiceEnv,
    RandomCornerEnv,
    ObjectDeliveryEnv,
)
from cocogrid.common.abstraction import RoomAbstraction
from cocogrid.tasks import extract_feature_from_mission_text, COLOR_NAMES, OBJECT_NAMES
from cocogrid.common.observation import ObservationSpecification
from cocogrid.common.entity import (
    get_color_id,
    ObjectEnum,
    COLOR_TO_IDX,
)
from cocogrid.common.multitask import MultiTaskEnv
import numpy as np
import gymnasium as gym
import minigrid.envs as envs
from minigrid.minigrid_env import MiniGridEnv

from gcrs.goal.goal_wrapper import GoalWrapper, GoalObserver
from gcrs.goal.planner import AbstractState, Observation, AStarPlanner


def goal_task_getter(obs: Observation, abstract: RoomAbstraction, env: gym.Env):
    """Get an abstract state evaluator to signal when a goal (or lava) has been reached."""

    def eval_state(abstract: RoomAbstraction):
        for entity_a, entity_b in abstract.nears:
            if (
                entity_a == 0
                and abstract.objects[entity_b - 1][0] == RoomAbstraction.GOAL_IDX
            ):
                return 1, True
        return 0, False

    return eval_state


def pickup_object_task_getter(
    obs: Observation, abstract: RoomAbstraction, env: gym.Env, strict=False
):
    """Get an abstract state evaluator to signal when the task-specified object has been picked up."""
    task = env.unwrapped.task
    color_name = extract_feature_from_mission_text(task, COLOR_NAMES)
    color_idx = get_color_id(color_name)
    type_name = extract_feature_from_mission_text(task, OBJECT_NAMES)
    type_idx = ObjectEnum.get_id(type_name)

    def eval_state(abstract: RoomAbstraction):
        for obj_type, color, state in abstract.objects:
            # if is a held object
            if obj_type != RoomAbstraction.DOOR_IDX and state == 1:
                # if matches target
                if color == color_idx and obj_type == type_idx:
                    return 1, True
                elif strict:
                    # in strict mode, only allow picking up target
                    return -1, True
        return 0, False

    return eval_state


def pickup_object_strict_task_getter(
    obs: Observation, abstract: RoomAbstraction, env: gym.Env
):
    """Only the task-specified object can be picked up."""
    return pickup_object_task_getter(obs, abstract, env, strict=True)


def object_delivery_task_getter(
    obs: Observation,
    abstract: RoomAbstraction,
    env: gym.Env,
    return_to_center: bool = False,
):
    """Get an abstract state evaluator to signal when the specified object has been delivered to the goal."""
    task = env.unwrapped._task
    pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
    matches = re.search(pattern, task.description)

    if not matches:
        raise Exception(f"Task '{task}' does not meet specification for RandomObject")
    color = matches.group(1)
    color_idx = get_color_id(color)
    class_name = matches.group(2)
    class_idx = ObjectEnum.get_id(class_name)

    def eval_state(abstract: RoomAbstraction):
        target_indices = []
        goal_index = -1
        for idx, (obj_type, color, state) in enumerate(abstract.objects):
            if obj_type == class_idx and color == color_idx:
                target_indices.append(idx + 1)
            elif obj_type == RoomAbstraction.GOAL_IDX:
                goal_index = idx + 1

        for t_idx in target_indices:
            if (t_idx, goal_index) in abstract.nears:
                return 1, True
        return 0, False

    return eval_state


def infer_task_goal(obs, abstract, _env):
    goal_fn = COCOGRID_ABSTRACT_GOALS[type(_env.unwrapped.minigrid.unwrapped)]
    return goal_fn(obs, abstract, _env)


COCOGRID_ABSTRACT_GOALS: Dict[MiniGridEnv, Callable] = {
    envs.CrossingEnv: goal_task_getter,
    envs.DoorKeyEnv: goal_task_getter,
    envs.FourRoomsEnv: goal_task_getter,
    envs.EmptyEnv: goal_task_getter,
    envs.DistShiftEnv: goal_task_getter,
    envs.LavaGapEnv: goal_task_getter,
    envs.LockedRoomEnv: goal_task_getter,
    envs.MultiRoomEnv: goal_task_getter,
    UMazeEnv: goal_task_getter,
    HallwayChoiceEnv: goal_task_getter,
    RandomCornerEnv: goal_task_getter,
    envs.KeyCorridorEnv: pickup_object_task_getter,
    envs.UnlockPickupEnv: pickup_object_task_getter,
    envs.FetchEnv: pickup_object_strict_task_getter,
    ObjectDeliveryEnv: object_delivery_task_getter,
    MultiTaskEnv: infer_task_goal,
}

# there are 5 object types: ball, box, door, key, goal
object_type_one_hot = np.arange(len(RoomAbstraction.OBJECT_IDS))
# there are 6 object color: red, green, blue, magenta, yellow, grey
object_color_one_hot = np.arange(len(COLOR_TO_IDX))
# the door has a max state of 3
object_state_one_hot = np.arange(4)


# goal observation function to map the abstracted state into a observation representation (e.g. a vector)
def object_obs_fn(abstract: RoomAbstraction, env=None):
    """Observe the RoomAbstraction to get information about held object and nearby object."""
    held_idx = abstract.get_held_object()
    if held_idx == -1:
        held_type, held_color = -1, -1
    else:
        held_type, held_color = abstract.objects[held_idx][:2]
    near_idx = abstract.get_near_object()
    if near_idx == -1:
        near_type, near_color, near_state = -1, -1, -1
    else:
        near_type, near_color, near_state = abstract.objects[near_idx]
    if env is not None:
        if abstract._positions is not None and 0 <= near_idx < len(abstract.objects):
            near_pos = abstract._positions[near_idx + 1]
        else:
            near_pos = (-1, -1)
    else:
        near_pos = []
    return (held_type, held_color, near_type, near_color, near_state, *near_pos)


def object_obs_fn_one_hot(abstract: RoomAbstraction):
    """Observe the RoomAbstraction to get information about held object and nearby object, encoded as one-hot."""
    held_idx = abstract.get_held_object()
    if held_idx == -1:
        held_type, held_color = -1, -1
    else:
        held_type, held_color = abstract.objects[held_idx][:2]
    near_idx = abstract.get_near_object()
    if near_idx == -1:
        near_type, near_color, near_state = -1, -1, -1
    else:
        near_type, near_color, near_state = abstract.objects[near_idx]
    return (
        *ObservationSpecification.get_maybe_one_hot(held_type, object_type_one_hot),
        *ObservationSpecification.get_maybe_one_hot(held_color, object_color_one_hot),
        *ObservationSpecification.get_maybe_one_hot(near_type, object_type_one_hot),
        *ObservationSpecification.get_maybe_one_hot(near_color, object_color_one_hot),
        *ObservationSpecification.get_maybe_one_hot(near_state, object_state_one_hot),
    )


def get_room_abstraction_goal_wrapper(
    env: gym.Env,
    env_id: str,
    cls: Type[GoalWrapper] = GoalWrapper,
    one_hot=False,
    use_pos=False,
    observe_subgoal=True,
):
    """Construct a GridGoalWrapper based on the room abstraction

    Input:
    env (gym.Env): A CocoGrid gymnasium environment.
    env_id (str): The environment id used to create the environmnet.
    goal_cls (Type[GoalWrapper]): A subclass of GoalWrapper to manage abstractions, rewards, and observations.
    observe_subgoal (bool): Can be set to false to make the agent unaware of the plan (e.g. as a baseline)
    """

    # abstraction function to map continuous to discrete
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return RoomAbstraction.from_cocogrid_state(state)

    minigrid_env = env.unwrapped.minigrid
    if observe_subgoal:
        if one_hot:
            obs_len = (
                2 * len(object_type_one_hot)
                + 2 * len(object_color_one_hot)
                + len(object_state_one_hot)
            )
            low = np.zeros(obs_len)
            high = np.ones(obs_len)
            observer = GoalObserver(object_obs_fn_one_hot, low, high)
        else:
            low = [0, 0, 0, 0, 0, 0, 0]
            high = [
                RoomAbstraction.GOAL_IDX,
                len(COLOR_NAMES),
                RoomAbstraction.GOAL_IDX,
                len(COLOR_NAMES),
                3,
                minigrid_env.grid.width,
                minigrid_env.grid.height,
            ]
            obs_len = 7 if use_pos else 5
            obs_func = partial(object_obs_fn, env=env.unwrapped if use_pos else None)
            observer = GoalObserver(obs_func, low[:obs_len], high[:obs_len])
    else:
        observer = GoalObserver(lambda abstract: [], [], [])

    # get the task-specific goal function for this environment
    goal_fn = COCOGRID_ABSTRACT_GOALS[type(minigrid_env)]

    # define the planner that plans over a network of abstract states, give a particular task
    def task_edge_getter(obs: Observation, abstract: RoomAbstraction, env: gym.Env):
        eval_state = goal_fn(obs, abstract, env)

        def get_edges(abstract: AbstractState):
            for neighbor in abstract.get_possible_actions().values():
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

    return cls(env, abstraction_fn, planner, observer)

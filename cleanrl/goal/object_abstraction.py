from functools import partial
import re
from typing import Callable, Dict, Type
import gymnasium as gym
import minigrid.envs as envs
from minigrid.minigrid_env import MiniGridEnv
from minimujo.custom_minigrid import UMazeEnv, HallwayChoiceEnv, RandomCornerEnv, RandomObjectsEnv
from minimujo.state.object_abstraction import ObjectAbstraction
from minimujo.state.goal_wrapper import GoalWrapper, AbstractState, Observation, DjikstraBackwardsPlanner, GoalObserver, AStarPlanner
from minimujo.state.tasks import extract_feature_from_mission_text, COLOR_NAMES, OBJECT_NAMES
from minimujo.state.observation import ObservationSpecification
from minimujo.entities import get_color_id, ObjectEnum, get_color_name, COLOR_TO_IDX
from minimujo.utils.visualize.drawing import get_camera_bounds, draw_rectangle
from minimujo.color import get_color_rgba_255
from minimujo.multitask import MultiTaskEnv
from minimujo.entities import OBJECT_NAMES
from minimujo.state.minimujo_state import MinimujoState
import numpy as np
import cv2

def goal_task_getter(obs: Observation, abstract: ObjectAbstraction, env: gym.Env):
    def eval_state(abstract: ObjectAbstraction):
        for entity_a, entity_b in abstract.nears:
            if entity_a == 0 and abstract.objects[entity_b-1][0] == ObjectAbstraction.GOAL_IDX:
                return 1, True
        return 0, False
    return eval_state

def pickup_object_task_getter(obs: Observation, abstract: ObjectAbstraction, env: gym.Env, strict=False):
    task = env.unwrapped.task
    color_name = extract_feature_from_mission_text(task, COLOR_NAMES)
    color_idx = get_color_id(color_name)
    type_name = extract_feature_from_mission_text(task, OBJECT_NAMES)
    type_idx = ObjectEnum.get_id(type_name)

    def eval_state(abstract: ObjectAbstraction):
        for (obj_type, color, state) in abstract.objects:
            # if is a held object
            if obj_type != ObjectAbstraction.DOOR_IDX and state == 1:
                # if matches target
                if color == color_idx and obj_type == type_idx:
                    return 1, True
                elif strict:
                    # in strict mode, only allow picking up target
                    return -1, True
        return 0, False
    return eval_state

def pickup_object_strict_task_getter(obs: Observation, abstract: ObjectAbstraction, env: gym.Env):
    return pickup_object_task_getter(obs, abstract, env, strict=True)

def random_object_task_getter(obs: Observation, abstract: ObjectAbstraction, env: gym.Env, return_to_center: bool = False):
    task = env.unwrapped._task
    pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
    matches = re.search(pattern, task.description)

    if not matches:
        raise Exception(f"Task '{task}' does not meet specification for RandomObject")
    color = matches.group(1)
    color_idx = get_color_id(color)
    class_name = matches.group(2)
    class_idx = ObjectEnum.get_id(class_name)
    # this abstraction has no notion of x, y. it will need to look and find the goal itself
    # x = int(matches.group(3))
    # y = int(matches.group(4))

    def eval_state(abstract: ObjectAbstraction):
        target_indices = []
        goal_index = -1
        for idx, (obj_type, color, state) in enumerate(abstract.objects):
            if obj_type == class_idx and color == color_idx:
                target_indices.append(idx+1)
            elif obj_type == ObjectAbstraction.GOAL_IDX:
                goal_index = idx+1

        for t_idx in target_indices:
            if (t_idx, goal_index) in abstract.nears:
                return 1, True
        return 0, False

    return eval_state

def infer_task_goal(obs, abstract, _env):
    goal_fn = MINIMUJO_ABSTRACT_GOALS[type(_env.unwrapped.minigrid.unwrapped)]
    return goal_fn(obs, abstract, _env)

MINIMUJO_ABSTRACT_GOALS: Dict[MiniGridEnv, Callable] = {
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
    # envs.PutNearEnv: get_put_near_task
    RandomObjectsEnv: random_object_task_getter,
    MultiTaskEnv: infer_task_goal,
}

# there are 5 object types: ball, box, door, key, goal
object_type_one_hot = np.arange(len(ObjectAbstraction.OBJECT_IDS))
# there are 6 object color: red, green, blue, magenta, yellow, grey
object_color_one_hot = np.arange(len(COLOR_TO_IDX))
# the door has a max state of 3
object_state_one_hot = np.arange(4)

# goal observation function to map the abstracted state into a observation representation (e.g. a vector)
def object_obs_fn(abstract: ObjectAbstraction, env=None):
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
            near_pos = abstract._positions[near_idx+1]
        else:
            near_pos = (-1, -1)
    else:
        near_pos = []
    return (held_type, held_color, near_type, near_color, near_state, *near_pos)

def object_obs_fn_one_hot(abstract: ObjectAbstraction):
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
        *ObservationSpecification.get_maybe_one_hot(near_state, object_state_one_hot)
    )

def get_object_abstraction_goal_wrapper(env: gym.Env, env_id: str, cls: Type[GoalWrapper]=GoalWrapper, one_hot=False, use_pos=False, observe_subgoal=True):
    # abstraction function to map continuous to discrete
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return ObjectAbstraction.from_minimujo_state(state)
    
    minigrid_env = env.unwrapped.minigrid
    if observe_subgoal:
        if one_hot:
            obs_len = 2 * len(object_type_one_hot) + 2 * len(object_color_one_hot) + len(object_state_one_hot)
            low = np.zeros(obs_len)
            high = np.ones(obs_len)
            observer = GoalObserver(object_obs_fn_one_hot, low, high)
        else:
            low = [0, 0, 0, 0, 0, 0, 0]
            high = [ObjectAbstraction.GOAL_IDX, len(COLOR_NAMES), ObjectAbstraction.GOAL_IDX, len(COLOR_NAMES), 3, minigrid_env.grid.width, minigrid_env.grid.height]
            obs_len = 7 if use_pos else 5
            obs_func = partial(object_obs_fn, env=env.unwrapped if use_pos else None)
            observer = GoalObserver(obs_func, low[:obs_len], high[:obs_len])
    else:
        observer = GoalObserver(lambda abstract: [], [], [])

    # get the task-specific goal function for this environment
    goal_fn = MINIMUJO_ABSTRACT_GOALS[type(minigrid_env)]

    # define the planner that plans over a network of abstract states, give a particular task
    def task_edge_getter(obs: Observation, abstract: ObjectAbstraction, env: gym.Env):
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

class GoalStateObserver(GoalObserver):

    def __init__(self, *args, env=None):
        super().__init__(*args)
        self.env = env

    def observe(self, goal: AbstractState) -> np.ndarray:
        return self.goal_observation_fn(goal, self.env)

class ObjectGoalWrapper(GoalWrapper):

    def __init__(self, *args, gamma=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def render(self):
        image = super().render()
        if image is not None:
            bounds = get_camera_bounds(self, norm_scale=True)
            # pos = self.subgoal.walker_pos

            (held_type, held_color, near_type, near_color, near_state) = object_obs_fn(self.subgoal)

            if held_type == -1:
                held_color = (255,128,40)
                held_text = "Not holding."
            else:
                held_name = ObjectAbstraction.OBJECT_IDS[held_type]
                held_color_name = get_color_name(held_color)
                held_color = tuple(get_color_rgba_255(held_color_name)[:3])
                held_text = f"Hold {held_color_name} {held_name}."

            if near_type == -1:
                near_color = (255,128,40)
                near_text = "Not near."
            else:
                near_name = ObjectAbstraction.OBJECT_IDS[near_type]
                near_color_name = get_color_name(near_color)
                near_color = tuple(get_color_rgba_255(near_color_name)[:3])
                door_state = ''
                if near_type == ObjectAbstraction.DOOR_IDX:
                    door_state = ['open ', 'closed ', 'locked ', 'locked '][near_state]
                near_text = f"Go near {door_state}{near_color_name} {near_name}."
                
            # draw_rectangle(image, bounds, (pos[0], -pos[1]), (pos[0] + 1, -pos[1]-1), color, 4)

            cv2.putText(image, held_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, held_color, 1, lineType=2)
            cv2.putText(image, near_text, (image.shape[0] // 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, near_color, 1, lineType=2)
        else:
            print("Goal image was none")
            print("super type", type(self.env))
        return image

class PBRSObjectGoalWrapper(ObjectGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        return prev_cost - self._planner.cost
    
class SubgoalObjectGoalWrapper(ObjectGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 1 if self.abstract_state == prev_subgoal else -1
    
class SubgoalCostObjectGoalWrapper(ObjectGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        if prev_abstract == self.abstract_state:
            return -1
        return 0 if self.abstract_state == prev_subgoal else -400
    
class SubgoalLargePenaltyObjectGoalWrapper(ObjectGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 100 if self.abstract_state == prev_subgoal else -400
    
class SubgoalRewardOnlyObjectGoalWrapper(ObjectGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return int(self.abstract_state == prev_subgoal)
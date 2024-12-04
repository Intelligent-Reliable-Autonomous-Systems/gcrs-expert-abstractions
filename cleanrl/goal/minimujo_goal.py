import re
from typing import Callable, Dict
import minigrid.envs as envs
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import gymnasium as gym

from minimujo.state.goal_wrapper import GoalWrapper, AbstractState, Observation, DjikstraBackwardsPlanner, GoalObserver
from minimujo.state.grid_abstraction import GridAbstraction
from minimujo.entities import get_color_id, ObjectEnum, get_color_name
from minimujo.state.tasks import extract_feature_from_mission_text, COLOR_NAMES, OBJECT_NAMES
from minimujo.multitask import MultiTaskEnv
from minimujo.custom_minigrid import UMazeEnv, HallwayChoiceEnv, RandomCornerEnv, RandomObjectsEnv
from minimujo.utils.visualize.drawing import get_camera_bounds, draw_rectangle
from minimujo.color import get_color_rgba_255

def goal_task_getter(obs, abstract, _env):
    goal_idx = np.where(abstract.grid == GridAbstraction.GRID_GOAL)
    goal_x, goal_y = goal_idx[0][0], goal_idx[1][0]
    return GridAbstraction(abstract.grid, (goal_x, goal_y), abstract.objects)

def random_object_task_getter(obs, abstract, _env):
    task = _env.unwrapped._task
    pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
    matches = re.search(pattern, task.description)

    if not matches:
        raise Exception(f"Task '{task}' does not meet specification for RandomObject")
    color = matches.group(1)
    color_idx = get_color_id(color)
    class_name = matches.group(2)
    class_idx = ObjectEnum.get_id(class_name)
    x = int(matches.group(3))
    y = int(matches.group(4))
    objects = abstract.objects.copy()
    for idx, obj in enumerate(abstract.objects):
        if obj[0] == class_idx and obj[3] == color_idx:
            objects[idx] = (obj[0], x, y, obj[3], 0)

    return GridAbstraction(abstract.grid, (x, y), objects)

def pickup_object_task_getter(obs, abstract, _env):
    task = _env.unwrapped.task
    color_name = extract_feature_from_mission_text(task, COLOR_NAMES)
    color_idx = get_color_id(color_name)
    type_name = extract_feature_from_mission_text(task, OBJECT_NAMES)
    type_idx = ObjectEnum.get_id(type_name)

    agent_x, agent_y = -1, -1
    objects = abstract.objects.copy()
    for idx, obj in enumerate(abstract.objects):
        if obj[0] == type_idx and obj[3] == color_idx:
            objects[idx] = (*obj[:4], 1)
            agent_x, agent_y = obj[1], obj[2]
    assert agent_x >= 0, f"Object {color_name} {type_name} in task could not be found"

    return GridAbstraction(abstract.grid, (agent_x, agent_y), objects)

def infer_task_goal(obs, abstract, _env):
    goal_fn = MINIMUJO_ABSTRACT_GOALS[type(_env.unwrapped.minigrid.unwrapped)]
    return goal_fn(obs, abstract, _env)

MINIMUJO_ABSTRACT_GOALS: Dict[MiniGridEnv, Callable] = {
    envs.FetchEnv: pickup_object_task_getter,
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
    RandomObjectsEnv: random_object_task_getter,
    RandomCornerEnv: goal_task_getter
    # envs.PutNearEnv: get_put_near_task
}

def get_minimujo_goal_wrapper(env: gym.Env, env_id: str, cls=GoalWrapper):
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_minimujo_state(state)
    planner = DjikstraBackwardsPlanner(GridAbstraction.backward_neighbor_edges)
    def goal_obs_fn(abstract):
        held_type, held_color = -1, -1
        if abstract._held_object >= 0:
            held = abstract.objects[abstract._held_object]
            held_type, held_color = held[0], held[3]
        return (*abstract.walker_pos, held_type, held_color)
    minigrid_env = env.unwrapped.minigrid
    low = [0, 0, 0, 0]
    high = [minigrid_env.grid.width, minigrid_env.grid.height, len(OBJECT_NAMES), len(COLOR_NAMES)]
    observer = GoalObserver(goal_obs_fn, low, high)

    goal_fn = MINIMUJO_ABSTRACT_GOALS[type(minigrid_env)]

    return cls(env, abstraction_fn, goal_fn, planner, observer)

class GridGoalWrapper(GoalWrapper):

    def render(self):
        image = super().render()
        if image is not None:
            bounds = get_camera_bounds(self, norm_scale=True)
            pos = self.subgoal.walker_pos

            if self.subgoal._held_object == -1:
                color = np.array([255,128,40])
            else:
                obj = self.subgoal.objects[self.subgoal._held_object]
                color_name = get_color_name(obj[3])
                color = get_color_rgba_255(color_name)[:3]
                
            draw_rectangle(image, bounds, (pos[0], -pos[1]), (pos[0] + 1, -pos[1]-1), color, 4)
        else:
            print("Goal image was none")
            print("super type", type(self.env))
        return image

class PBRSGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        return prev_cost - self._planner.cost
    
class PBRSDenseGoalWrapper(GridGoalWrapper):

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        self.prev_total_dist = self._planner.cost + dist
        return output

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        diff = self.prev_total_dist - total_dist
        self.prev_total_dist = total_dist
        return diff
    
class DistanceCostGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        return -total_dist
    
class DistanceCostFractionGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        return -(total_dist / self._initial_plan_cost)
    
class SubgoalDistanceGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        return -dist

class SubgoalGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 1 if self.abstract_state == prev_subgoal else -1
    
class SubgoalLargePenaltyGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 100 if self.abstract_state == prev_subgoal else -400
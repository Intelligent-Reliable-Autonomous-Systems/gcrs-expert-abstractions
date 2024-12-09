import re
from typing import Callable, Dict, Type
import minigrid.envs as envs
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import gymnasium as gym

from minimujo.state.goal_wrapper import GoalWrapper, AbstractState, Observation, DjikstraBackwardsPlanner, GoalObserver, AStarPlanner
from minimujo.state.grid_abstraction import GridAbstraction
from minimujo.entities import get_color_id, ObjectEnum, get_color_name
from minimujo.state.tasks import extract_feature_from_mission_text, COLOR_NAMES, OBJECT_NAMES
from minimujo.multitask import MultiTaskEnv
from minimujo.custom_minigrid import UMazeEnv, HallwayChoiceEnv, RandomCornerEnv, RandomObjectsEnv
from minimujo.utils.visualize.drawing import get_camera_bounds, draw_rectangle
from minimujo.color import get_color_rgba_255

def goal_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env):
    def eval_state(abstract: GridAbstraction):
        grid = abstract.walker_grid_cell
        if grid == GridAbstraction.GRID_GOAL:
            return 1, True
        elif grid == GridAbstraction.GRID_LAVA:
            return -1, True
        return 0, False
    return eval_state


def random_object_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env):
    task = env.unwrapped._task
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
    target_object = (class_idx, x, y, color_idx, 0)

    def eval_state(abstract: GridAbstraction):
        for obj in abstract.objects:
            if obj == target_object:
                return 1, True
        return 0, False

    return eval_state

def pickup_object_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env, strict=False):
    task = env.unwrapped.task
    color_name = extract_feature_from_mission_text(task, COLOR_NAMES)
    color_idx = get_color_id(color_name)
    type_name = extract_feature_from_mission_text(task, OBJECT_NAMES)
    type_idx = ObjectEnum.get_id(type_name)

    def eval_state(abstract: GridAbstraction):
        for (obj_type, _, _, color, state) in abstract.objects:
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

def pickup_object_strict_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env):
    return pickup_object_task_getter(obs, abstract, env, strict=True)

def infer_task_goal(obs, abstract, _env):
    goal_fn = MINIMUJO_ABSTRACT_GOALS[type(_env.unwrapped.minigrid.unwrapped)]
    return goal_fn(obs, abstract, _env)

MINIMUJO_ABSTRACT_GOALS: Dict[MiniGridEnv, Callable] = {
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
    RandomObjectsEnv: random_object_task_getter,
    RandomCornerEnv: goal_task_getter
    # envs.PutNearEnv: get_put_near_task
}

def get_minimujo_goal_wrapper(env: gym.Env, env_id: str, cls: Type[GoalWrapper]=GoalWrapper):
    # abstraction function to map continuous to discrete
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_minimujo_state(state)
    
    # goal observation function to map the abstracted state into a observation representation (e.g. a vector)
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

    # get the task-specific goal function for this environment
    goal_fn = MINIMUJO_ABSTRACT_GOALS[type(minigrid_env)]

    # define the planner that plans over a network of abstract states, give a particular task
    def task_edge_getter(obs: Observation, abstract: AbstractState, env: gym.Env):
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

    return cls(env, abstraction_fn, planner, observer)

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
import re
from typing import Callable, Dict, Type
import minigrid.envs as envs
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import gymnasium as gym

from minimujo.state.goal_wrapper import GoalWrapper, AbstractState, Observation, DjikstraBackwardsPlanner, GoalObserver, AStarPlanner
from minimujo.state.minimujo_state import MinimujoState
from minimujo.state.grid_abstraction import GridAbstraction
from minimujo.entities import get_color_id, ObjectEnum, get_color_name
from minimujo.state.tasks import extract_feature_from_mission_text, COLOR_NAMES, OBJECT_NAMES
from minimujo.multitask import MultiTaskEnv
from minimujo.custom_minigrid import UMazeEnv, HallwayChoiceEnv, RandomCornerEnv, RandomObjectsEnv
from minimujo.minigrid.doorkeycrossing import DoorKeyCrossingEnv
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


def random_object_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env, return_to_center: bool = False):
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

    if return_to_center:
        def eval_state(abstract: GridAbstraction):
            if abstract.walker_pos == (2,2):
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

def random_object_center_task_getter(obs: Observation, abstract: GridAbstraction, env: gym.Env):
    return random_object_task_getter(obs, abstract, env, return_to_center=True)

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
    RandomObjectsEnv: random_object_center_task_getter,
    RandomCornerEnv: goal_task_getter,
    DoorKeyCrossingEnv: goal_task_getter
    # envs.PutNearEnv: get_put_near_task
}

def get_minimujo_goal_wrapper(env: gym.Env, env_id: str, cls: Type[GoalWrapper]=GoalWrapper, snap_held=True, observe_subgoal=True):
    # abstraction function to map continuous to discrete
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_minimujo_state(state, snap_held_to_agent=snap_held)
    
    minigrid_env = env.unwrapped.minigrid
    # goal observation function to map the abstracted state into a observation representation (e.g. a vector)
    if observe_subgoal:
        def goal_obs_fn(abstract):
            held_type, held_color = -1, -1
            if abstract._held_object >= 0:
                held = abstract.objects[abstract._held_object]
                held_type, held_color = held[0], held[3]
            return (*abstract.walker_pos, held_type, held_color)
        low = [0, 0, 0, 0]
        high = [minigrid_env.grid.width, minigrid_env.grid.height, len(OBJECT_NAMES), len(COLOR_NAMES)]
        observer = GoalObserver(goal_obs_fn, low, high)
    else:
        observer = GoalObserver(lambda abstract: [], [], [])

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

    def __init__(self, *args, gamma=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

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

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        prev_potential = -prev_cost
        new_potential = - self._planner.cost
        # if term:
        #     new_potential = 0
        # return self.gamma * new_potential - prev_potential
        return new_potential - prev_potential
    
class PBRSDenseGoalWrapper(GridGoalWrapper):

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        self.prev_total_dist = self._planner.cost + dist
        return output

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        diff = self.prev_total_dist - total_dist
        self.prev_total_dist = total_dist
        return diff
    
class PBRSDenseGoalWrapperV2(GridGoalWrapper):

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.distance_between_states(curr_state, self.abstract_state, self.subgoal)
        self.prev_total_dist = self._planner.cost + dist
        return output

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.distance_between_states(curr_state, self.abstract_state, self.subgoal)
        print('dist', dist)
        total_dist = self._planner.cost + dist
        diff = self.prev_total_dist - total_dist
        self.prev_total_dist = total_dist
        return diff
    
class PBRSDenseGoalWrapperV4(GridGoalWrapper):

    def __init__(self, 
                 *args, 
                 dist_weight=0.5, 
                 vel_weight=0.5, 
                 activate_key_reward=False, 
                 activate_pickup_reward=False, 
                 linear_potential=True,
                 zero_term_potential=True,
                 gamma_override=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_weight = dist_weight
        self.vel_weight = vel_weight
        self.activate_key_reward = activate_key_reward
        self.activate_pickup_reward = activate_pickup_reward
        self.linear_potential = linear_potential
        self.zero_term_potential = zero_term_potential
        self.gamma = gamma_override

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        curr_state = self.env.unwrapped.state
        dist = self.dense_potential(curr_state, self.abstract_state, self.subgoal)
        self.prev_total_dist = self._planner.cost + dist
        return output

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        curr_state = self.env.unwrapped.state
        dist_to_subgoal = self.dense_potential(curr_state, self.abstract_state, self.subgoal)
        new_total_dist = self._planner.cost + dist_to_subgoal

        old_potential = - self.prev_total_dist
        new_potential = - new_total_dist
        if self.zero_term_potential and term:
            new_potential = 0

        diff_in_potential =  self.gamma * new_potential - old_potential

        self.prev_total_dist = new_total_dist

        return diff_in_potential

    def get_index_of_object_to_pick_up(self, curr_abstract: GridAbstraction, subgoal_abstract: GridAbstraction):
        for idx, ((curr_obj_type, _, _, _, curr_obj_state), (_, _, _, _, subgoal_obj_state)) in enumerate(zip(curr_abstract.objects, subgoal_abstract.objects)):
            # everything but door is holdable
            is_holdable = curr_obj_type != GridAbstraction.DOOR_IDX 
            is_curr_holding = curr_obj_state > 0
            is_subgoal_holding = subgoal_obj_state > 0
            needs_to_pick_up = is_holdable and not is_curr_holding and is_subgoal_holding
            if needs_to_pick_up:
                return idx
        return -1
    
    def dense_potential(self, minimujo_state: MinimujoState, curr_abstract: GridAbstraction, subgoal_abstract: GridAbstraction):
        """Compute a continuous 'distance' from the target abstract state based on distance and velocity"""

        # get agent position and velocity
        curr_pos = minimujo_state.pose[MinimujoState.POSE_IDX_POS:MinimujoState.POSE_IDX_POS+2] / minimujo_state.xy_scale
        curr_vel = minimujo_state.pose[MinimujoState.POSE_IDX_VEL:MinimujoState.POSE_IDX_VEL+2]
        # assign target to current by default
        targ_pos = curr_pos

        # agent needs to move to the boundary of the next subogal
        tx, ty = subgoal_abstract.walker_pos
        # we are getting the closest point on the boundary of the subgoal
        # from the current position, we want the closest point on the boundary of the subgoal
        # the subgoal boundary is defined by the four corners of the subgoal [tx, -(ty+1)], [tx+1, -ty]
        targ_pos = np.clip(curr_pos, [tx, -(ty+1)], [tx+1, -ty])

        if self.activate_pickup_reward:
            # check if an object needs to be picked up. if so, set the object's position as the target position
            pickup_idx = self.get_index_of_object_to_pick_up(curr_abstract, subgoal_abstract)
            if pickup_idx != -1:
                targ_pos = minimujo_state.objects[pickup_idx][MinimujoState.OBJECT_IDX_POS:MinimujoState.OBJECT_IDX_POS+2] / minimujo_state.xy_scale

        # compute the distance and velocity
        dist = np.linalg.norm(targ_pos - curr_pos)
        if dist == 0:
            return 0
        direction = (targ_pos - curr_pos) / dist
        velocity_in_target_direction = np.dot(curr_vel, direction)

        if self.linear_potential:
            # scale distance from maximum 1.5 to maximum 1
            distance_cost = min(1, 0.75 * dist)
            # scale velocity from range (-3, 3) to (0,1)
            velocity_cost = max(0, min(1, (3-velocity_in_target_direction) / 6))
        else:
            # distance should be near 1 when far away and near 0 when close
            distance_cost = 1 - np.exp(-2 * dist)
            # velocity should be near 1 when moving away from target and near 0 when moving fast towards it
            velocity_cost = min(1, np.exp(-0.9 * (velocity_in_target_direction + 1)))

        return self.dist_weight * distance_cost + self.vel_weight * velocity_cost
    
class DistanceCostGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        return -total_dist
    
class DistanceCostFractionGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        return -(total_dist / self._initial_plan_cost)
    
class SubgoalDistanceGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        return -dist

class SubgoalGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 1 if self.abstract_state == prev_subgoal else -1
    
class SubgoalLargePenaltyGoalWrapper(GridGoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float, term: bool) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 100 if self.abstract_state == prev_subgoal else -400
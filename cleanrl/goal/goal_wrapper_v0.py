from typing import Any, Dict, Optional, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from minimujo.state.grid_abstraction import GridAbstraction

class GridPositionGoalWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, term_on_reach=False, dense=False, env_id='Minimujo-UMaze-v0') -> None:
        super().__init__(env)
        if env_id == 'Minimujo-UMaze-v0':
            self.goal_seq = get_umaze_goals
            self.goal_obs_func = lambda abstract: abstract.walker_pos
            goal_low = [0,0]
            goal_high = [5,5]
        elif env_id == 'Minimujo-RandomObject-v0':
            self.goal_seq = get_randobj_goals
            self.goal_obs_func = lambda abstract: (*abstract.walker_pos, *(abstract.objects[0][1:3]), abstract.objects[0][4])
            goal_low = [0, 0, 0, 0, 0]
            goal_high = [5, 5, 5, 5, 1]
        else:
            raise Exception(f'There is no goal specification for env {env_id}')
        self.term_on_reach = term_on_reach
        self.dense = dense

        base_obs_space = self.env.unwrapped.observation_space
        assert isinstance(base_obs_space, gym.spaces.Box) and len(base_obs_space.shape) == 1
        new_low = np.concatenate([base_obs_space.low, goal_low], axis=None)
        new_high = np.concatenate([base_obs_space.high, goal_high], axis=None)
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=base_obs_space.dtype)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(*args, **kwargs)

        abstract_state = GridAbstraction.from_minimujo_state(self.env.unwrapped.state)
        self.goal_path = self.goal_seq(abstract_state)
        self.off_path_length = len(self.goal_path) + 1
        goal = self.goal_path[0]

        self.prev_state = abstract_state
        self.prev_goal = goal

        return np.concatenate([obs, self.goal_obs_func(goal)]), info

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        
        curr_state = self.env.unwrapped.state
        abstract_state = GridAbstraction.from_minimujo_state(curr_state)
        # temporarily, hard code to UMaze
        # TODO generalize planning

        # reward for subgoals
        goal_idx = (len(self.goal_path) - self.goal_path.index(self.prev_goal) - 1) if self.prev_goal in self.goal_path else self.off_path_length
        if self.dense:
            dist = GridAbstraction.grid_distance_from_state(self.prev_goal.walker_pos, curr_state)
            rew = -(goal_idx + dist) / len(self.goal_path)
            rew = min(rew, -0.1 / len(self.goal_path))
            # print(dist, self.prev_goal, curr_state.pose[:2] / curr_state.xy_scale)
        else:
            if abstract_state != self.prev_state:
                if abstract_state == self.prev_goal:
                    rew += 1
                else:
                    rew -= 1

        new_goal = self.goal_seq(abstract_state)[0]
        self.prev_state = abstract_state
        self.prev_goal = new_goal

        info['goal'] = new_goal
        info['num_subgoals'] = goal_idx
        info['frac_subgoals'] = goal_idx / len(self.goal_path)
        # if len(new_goal.objects) > 0:
        #     target_object_pos = self.goal_path[-1].objects[0][1:3]
        #     curr_object_pos = 

        # if self.term_on_reach and pos == new_goal:
        #     term = True
        # print(self.goal_obs_func(new_goal), rew, goal_idx, dist)
        return np.concatenate([obs, self.goal_obs_func(new_goal)]), rew, term, trunc, info
    
def get_umaze_goals(abstract_state):
    pos = abstract_state.walker_pos
    if pos == (1,1):
        return [abstract_state]
    pos_actions = {
        (2,1): GridAbstraction.ACTION_LEFT,
        (3,1): GridAbstraction.ACTION_LEFT,
        (3,2): GridAbstraction.ACTION_UP,
        (3,3): GridAbstraction.ACTION_UP,
        (2,3): GridAbstraction.ACTION_RIGHT,
        (1,3): GridAbstraction.ACTION_RIGHT
    }
    goal_sequence = []
    while pos != (1,1):
        if pos in pos_actions:
            abstract_state = abstract_state.do_action(pos_actions[pos])
            pos = abstract_state.walker_pos
            goal_sequence.append(abstract_state)
    return goal_sequence

def get_randobj_goals(abstract_state):
    def next_state(abstract_state):
        obj = abstract_state.objects[0]
        _, obj_x, obj_y, _, held = obj
        walker_x, walker_y = abstract_state.walker_pos
        goal_idx = np.where(abstract_state.grid == 2)
        goal_x, goal_y = goal_idx[0][0], goal_idx[1][0]
        if held:
            if goal_x != walker_x:
                # move horizontal towards goal w/ object
                if goal_x < walker_x:
                    return abstract_state.do_action(GridAbstraction.ACTION_LEFT)
                else:
                    return abstract_state.do_action(GridAbstraction.ACTION_RIGHT)
            if goal_y != walker_y:
                # move vertical towards goal w/ object
                if goal_y < walker_y:
                    return abstract_state.do_action(GridAbstraction.ACTION_UP)
                else:
                    return abstract_state.do_action(GridAbstraction.ACTION_DOWN)
            # else is at goal. release object
            return abstract_state.do_action(GridAbstraction.ACTION_GRAB)
        # else move towards object
        if walker_x != obj_x:
            if obj_x < walker_x:
                return abstract_state.do_action(GridAbstraction.ACTION_LEFT)
            else:
                return abstract_state.do_action(GridAbstraction.ACTION_RIGHT)
        if walker_y != obj_y:
            if obj_y < walker_y:
                return abstract_state.do_action(GridAbstraction.ACTION_UP)
            else:
                return abstract_state.do_action(GridAbstraction.ACTION_DOWN)
        if obj_x == goal_x and obj_y == goal_y:
            return None
        # else grab object
        return abstract_state.do_action(GridAbstraction.ACTION_GRAB)
    
    state = abstract_state
    if next_state(state) is None:
        return [state]
    goal_seq = []
    while state is not None:
        state = next_state(state)
        if state is None:
            break
        goal_seq.append(state)
    return goal_seq

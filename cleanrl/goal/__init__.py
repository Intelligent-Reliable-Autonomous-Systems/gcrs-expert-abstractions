# from minimujo.state.grid_abstraction import get_minimujo_goal_wrapper
from cleanrl.goal.minimujo_goal import *

goal_wrappers = {
    'goal-only': GoalWrapper,
    'pbrs': PBRSGoalWrapper,
    'pbrs-dense': PBRSDenseGoalWrapper,
    'distance-cost': DistanceCostGoalWrapper,
    'subgoal': SubgoalGoalWrapper,
    'subgoal-distance': SubgoalDistanceGoalWrapper,
    'subgoal-penalty': SubgoalLargePenaltyGoalWrapper
}

def wrap_env_with_goal(env, env_id, goal_version):
    if goal_version == 'no-goal':
        return env
    
    if 'Minimujo' in env_id:
        if goal_version in goal_wrappers:
            goal_cls = goal_wrappers[goal_version]
            return get_minimujo_goal_wrapper(env, env_id, goal_cls)
        
        print(f"WARNING: using legacy wrappers for goal_version {goal_version}")

        # legacy wrappers
        if goal_version == 'dense-v0':
            from cleanrl.goal.goal_wrapper_v0 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, dense=True, env_id=env_id)
        elif goal_version == 'dense-v1':
            from cleanrl.goal.goal_wrapper_v1 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, dense=True, env_id=env_id)
        elif goal_version == 'dense-v2':
            from cleanrl.goal.goal_wrapper_v2 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, dense=True, env_id=env_id)
        elif goal_version == 'dense-v3':
            from cleanrl.goal.goal_wrapper_v3 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, dense=True, env_id=env_id)
        elif goal_version == 'final-no-reward':
            from cleanrl.goal.final_goal_no_reward import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, dense=True, env_id=env_id)
        elif goal_version == 'option-v0':
            from cleanrl.goal.option_goal_wrapper import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, env_id=env_id)
        elif goal_version == 'option-v1':
            from cleanrl.goal.option_goal_wrapper_v1 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, env_id=env_id)
        elif goal_version == 'option-v2':
            from cleanrl.goal.option_goal_wrapper_v2 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, env_id=env_id)
        elif goal_version == 'option-v3':
            from cleanrl.goal.option_goal_wrapper_v3 import GridPositionGoalWrapper
            return GridPositionGoalWrapper(env, env_id=env_id)
    if "manipulation" in env_id:
        from cleanrl.goal.manipulation_goal import get_manipulator_goal_wrapper
        if goal_version in goal_wrappers:
            goal_cls = goal_wrappers[goal_version]
            return get_manipulator_goal_wrapper(env, env_id, goal_cls)
    raise Exception(f'goal version {goal_version} not valid in env {env_id}')
    
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.wrappers.human_rendering import HumanRendering
    import numpy as np
    from pygame import key
    import pygame
    import minimujo

    print_reward = True
    print_obs = False
    
    env_id = 'Minimujo-RandomObject-v0'
    env = gym.make(
        env_id,
        walker_type='box2d',
        xy_scale=3, 
        timesteps=600
    )
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480
    
    goal_env = wrap_env_with_goal(env, env_id, 'dense-v1')
    env = HumanRendering(goal_env)

    print('Controls: Move with WASD, grab with Space')

    def get_action():
        keys = key.get_pressed()
        up = 0
        right = 0
        grab = 0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            right += 1
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            right -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            up -= 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            up += 1
        if keys[pygame.K_n] or keys[pygame.K_SPACE]:
            grab = 1
        return np.array([grab, -up, right])

    obs, _ = env.reset()

    print(f'Env has observation space {env.unwrapped.observation_space} and action space {env.unwrapped.action_space}')
    print(f"Current task: {env.unwrapped.task}")

    num_steps = 0
    reward_sum = 0
    num_episodes = 0
    is_holding_reset = False
    while True:
        keys = key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print('Cumulative reward (to this point):', reward_sum)
            print('Manually terminated')
            break
        if keys[pygame.K_r] and not is_holding_reset:
            trunc = True
            is_holding_reset = True
        else:
            if not keys[pygame.K_r]:
                is_holding_reset = False
            action = env.unwrapped.action_space.sample()
            manual_action = get_action()
            action[:3] = manual_action

            obs, rew, term, trunc, info = env.step(action)
            reward_sum += rew
            num_steps += 1

            if print_reward:
                print('reward:', rew)

            if print_obs:
                print('obs:', obs)

            print(goal_env.prev_goal)
            
        if term or trunc:
            trunc_or_term = 'Truncated' if trunc else 'Terminated'
            print('Cumulative reward:', reward_sum)
            print(f'{trunc_or_term} after {num_steps} steps')
            num_episodes += 1
            env.reset()
            reward_sum = 0
            num_steps = 0
            term = trunc = False
            print(f"Current task: {env.unwrapped.task}")
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.wrappers.human_rendering import HumanRendering
    import numpy as np
    from pygame import key
    import pygame
    import minimujo
    # from goal import wrap_env_with_goal
    from cleanrl.goal import wrap_env_with_goal

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
    
    goal_env = wrap_env_with_goal(env, env_id, 'dense-v2')
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
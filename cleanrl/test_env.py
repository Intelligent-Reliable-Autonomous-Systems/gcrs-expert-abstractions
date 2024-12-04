if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.wrappers.human_rendering import HumanRendering
    import numpy as np
    from pygame import key
    import pygame
    import minimujo
    # from goal import wrap_env_with_goal
    from cleanrl.goal import wrap_env_with_goal

    import argparse
    from minimujo.utils.testing import add_minimujo_arguments, args_to_gym_env, get_pygame_action
    parser = argparse.ArgumentParser()
    add_minimujo_arguments(parser, env='Minimujo-RandomObject-v0', walker='box2d', scale=3, timesteps=600)
    parser.add_argument("--goal", "-g", type=str, default="pbrs", help="Goal wrapper version")
    parser.add_argument("--print-reward", action="store_true", help="Print reward")
    parser.add_argument("--print-obs", action="store_true", help="Print observation")
    parser.add_argument("--print-goal", action="store_true", help="Print goal")
    args = parser.parse_args()
    
    env_id = args.env
    env = args_to_gym_env(args)
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480

    goal_env = wrap_env_with_goal(env, env_id, args.goal)

    from minimujo.utils.logging import LoggingWrapper, MinimujoLogger
    from minimujo.utils.logging.tensorboard import MockSummaryWriter
    from minimujo.utils.logging.subgoals import SubgoalLogger
    env = LoggingWrapper(goal_env, MockSummaryWriter(), max_timesteps=600, raise_errors=True)
    # for logger in get_minimujo_heatmap_loggers(env, gamma=0.99):
    #     logger.label = f'{logging_params["prefix"]}_{logger.label}'.lstrip('_')
    #     env.subscribe_metric(logger)
    env.subscribe_metric(MinimujoLogger())
    env.subscribe_metric(SubgoalLogger())
    
    env = HumanRendering(env)

    print('Controls: Move with WASD, grab with Space')

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
            manual_action = get_pygame_action()
            action[:3] = manual_action

            obs, rew, term, trunc, info = env.step(action)
            reward_sum += rew
            num_steps += 1

            if args.print_reward:
                if rew != 0:
                    print('reward:', rew)

            if args.print_obs:
                print('obs:', obs)

            if args.print_goal:
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
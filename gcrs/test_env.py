"""This module is a utility to manually control an environment."""

if __name__ == "__main__":
    from gymnasium.wrappers.human_rendering import HumanRendering
    from pygame import key
    import pygame
    from gcrs.goal import wrap_env_with_goal

    import argparse
    from cocogrid.utils.testing import (
        add_cocogrid_arguments,
        args_to_gym_env,
        get_pygame_action,
    )

    parser = argparse.ArgumentParser()
    add_cocogrid_arguments(
        parser,
        env="cocogrid/DoorKey-6x6-v0",
        agent="box2d",
        scale=3,
        timesteps=600,
        include_seed=True,
    )
    parser.add_argument(
        "--goal", "-g", type=str, default="grid-pbrs", help="Goal wrapper version"
    )
    parser.add_argument("--gamma", type=float, default=1, help="discount factor")
    parser.add_argument("--print-reward", action="store_true", help="Print reward")
    parser.add_argument("--print-obs", action="store_true", help="Print observation")
    parser.add_argument("--print-goal", action="store_true", help="Print goal")
    parser.add_argument("--print-abstract", action="store_true", help="Print abstract")
    parser.add_argument(
        "--print-logs", action="store_true", help="Wrap in logger and print to console"
    )
    args = parser.parse_args()

    env_id = args.env
    env = args_to_gym_env(args)
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480

    env = goal_env = wrap_env_with_goal(env, env_id, args.goal, gamma=args.gamma)

    if args.print_logs:
        from cocogrid.utils.logging import LoggingWrapper, CocogridLogger
        from cocogrid.utils.logging.tensorboard import MockSummaryWriter
        from gcrs.utils.logging import SubgoalLogger

        env = LoggingWrapper(
            goal_env, summary_writer=MockSummaryWriter(), raise_errors=True
        )
        env.subscribe_metric(CocogridLogger())
        env.subscribe_metric(SubgoalLogger())

    env = HumanRendering(env)

    print("Controls: Move with WASD, grab with Space")

    obs, _ = env.reset()

    print(
        f"Env has observation space {env.observation_space} and action space {env.action_space}"
    )
    print(f"Current task: {env.unwrapped.task}")

    num_steps = 0
    reward_sum = 0
    cum_return = 0
    exp_gamma = 1
    num_episodes = 0
    is_holding_reset = False
    while True:
        keys = key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print("Cumulative reward (to this point):", reward_sum)
            print("Manually terminated")
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
            cum_return += rew * exp_gamma
            exp_gamma *= args.gamma
            num_steps += 1

            if args.print_reward:
                if rew != 0:
                    print("reward:", rew)

            if args.print_obs:
                print("obs:", obs.astype(int))

            if args.print_goal:
                print(goal_env.subgoal)

            if args.print_abstract:
                print(goal_env.abstract_state)

        if term or trunc:
            trunc_or_term = "Truncated" if trunc else "Terminated"
            if args.gamma == 1:
                print("Cumulative reward: {reward_sum:.2f}")
            else:
                print(
                    f"Cumulative reward: {reward_sum:.2f} | Cumulative return: {cum_return:.2f}"
                )
            print(f"{trunc_or_term} after {num_steps} steps")
            num_episodes += 1
            env.reset()
            reward_sum = 0
            cum_return = 0
            exp_gamma = 1
            num_steps = 0
            term = trunc = False
            print(f"Current task: {env.unwrapped.task}")

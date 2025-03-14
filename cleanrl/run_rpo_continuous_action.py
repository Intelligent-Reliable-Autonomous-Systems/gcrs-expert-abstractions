
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
from pygame import key
import pygame
import minimujo
# from goal import wrap_env_with_goal
from cleanrl.goal import wrap_env_with_goal
import torch
import argparse
from minimujo.utils.testing import add_minimujo_arguments, args_to_gym_env, get_pygame_action

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_minimujo_arguments(parser, env='Minimujo-RandomObject-v0', walker='box2d', scale=3, timesteps=600, include_seed=True)
    parser.add_argument("--goal", "-g", type=str, default="pbrs", help="Goal wrapper version")
    parser.add_argument("--gamma", type=float, default=1, help="discount factor")
    parser.add_argument("--checkpoint", '-c', type=str, default=None, help="if specified, load a model")
    parser.add_argument("--print-reward", action="store_true", help="Print reward")
    parser.add_argument("--print-obs", action="store_true", help="Print observation")
    parser.add_argument("--print-goal", action="store_true", help="Print goal")
    parser.add_argument("--print-abstract", action="store_true", help="Print abstract")
    parser.add_argument("--record", type=str, default=None, help="if specified will record a run then exit")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f'Using device {"cuda" if torch.cuda.is_available() and args.cuda else "cpu"}', device)
    
    env_id = args.env
    env = args_to_gym_env(args)
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480

    goal_env = wrap_env_with_goal(env, env_id, args.goal, gamma=args.gamma)

    env = gym.wrappers.FlattenObservation(goal_env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    # from minimujo.utils.logging import LoggingWrapper, MinimujoLogger
    # from minimujo.utils.logging.tensorboard import MockSummaryWriter
    # from minimujo.utils.logging.subgoals import SubgoalLogger
    # env = LoggingWrapper(goal_env, MockSummaryWriter(), max_timesteps=600, raise_errors=True)
    # for logger in get_minimujo_heatmap_loggers(env, gamma=0.99):
    #     logger.label = f'{logging_params["prefix"]}_{logger.label}'.lstrip('_')
    #     env.subscribe_metric(logger)
    # env.subscribe_metric(MinimujoLogger())
    # env.subscribe_metric(SubgoalLogger())
    
    if args.record is None:
        env = HumanRendering(env)
    else:
        env = RecordVideo(env, args.record, name_prefix="demo")
    
    rpo_alpha = 0.5
    agent = Agent(env, rpo_alpha).to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    agent.load_state_dict(torch.load(args.checkpoint))

    print('Controls: Move with WASD, grab with Space')

    obs, _ = env.reset()
    # breakpoint()

    print(f'Env has observation space {env.unwrapped.observation_space} and action space {env.unwrapped.action_space}')
    print(f"Current task: {env.unwrapped.task}")

    num_steps = 0
    reward_sum = 0
    cum_return = 0
    exp_gamma = 1
    num_episodes = 0
    is_holding_reset = False
    while True:
        if args.record is None:
            keys = key.get_pressed()
        else:
            keys = {}
        if keys.get(pygame.K_ESCAPE, False):
            print('Cumulative reward (to this point):', reward_sum)
            print('Manually terminated')
            break
        if keys.get(pygame.K_r, False) and not is_holding_reset:
            trunc = True
            is_holding_reset = True
        else:
            if not keys.get(pygame.K_r, False):
                is_holding_reset = False
            # action = env.unwrapped.action_space.sample()
            # manual_action = get_pygame_action()
            # action[:3] = manual_action

            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).to(torch.float32)
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)

            obs, rew, term, trunc, info = env.step(action.cpu().numpy()[0])
            reward_sum += rew
            cum_return += rew * exp_gamma
            exp_gamma *= args.gamma
            num_steps += 1

            if args.print_reward:
                if rew != 0:
                    print('reward:', rew)

            if args.print_obs:
                print('obs:', obs.astype(int))

            # print(obs[-26:-21], obs[-21:-15], obs[-15:-10], obs[-10:-4], obs[-4:])

            if args.print_goal:
                print(goal_env.subgoal)

            if args.print_abstract:
                print(goal_env.abstract_state)

            # print(len(goal_env._planner.plan))
            
        if term or trunc:
            trunc_or_term = 'Truncated' if trunc else 'Terminated'
            print('Cumulative reward:', reward_sum, 'cumulative return:', cum_return)
            print(f'{trunc_or_term} after {num_steps} steps')
            num_episodes += 1
            env.reset()
            reward_sum = 0
            cum_return = 0
            exp_gamma = 1
            num_steps = 0
            term = trunc = False
            print(f"Current task: {env.unwrapped.task}")
            if args.record is not None:
                env.close()
                break
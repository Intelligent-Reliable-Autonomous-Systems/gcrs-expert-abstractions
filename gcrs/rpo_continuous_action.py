"""Train a Robust Policy Optimization continuous-control agent. This implementation is from CleanRL."""

import os
import random
import time
from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from tensorboardX import SummaryWriter

from cocogrid.utils.logging import LoggingWrapper

from gcrs.env import get_environment_constructor_for_id, CocoGridEnv, is_cocogrid_env
from gcrs.model import RPOAgent


@dataclass
class Args:
    # environment parameters
    cocogrid: CocoGridEnv
    """Environment parameters for cocogrid"""

    goal_version: str = "dense-v0"
    """What experimental version of the wrapper to use"""
    reward_scale: float = 1
    """How much to scale the reward by"""
    reward_norm: bool = False
    """Whether to normalize reward"""
    eval_interval: int = 50000
    """How many steps in between evaluation"""
    eval_episodes: int = 10
    """How many episodes to run per eval"""

    exp_name: str = None
    """the name of this experiment"""
    group_name: str = None
    """the experiment group"""
    log_dir: str = None
    """the base dir where the log dir will go. default: 'runs/'"""
    video_dir: str = None
    """the base dir where the video dir will go. default: 'videos/'"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_tags: str = ""
    """Tags to be added to wandb"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""
    checkpoint_freq: int = 1_000_000
    """The number of steps between saving checkpoints"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    short_env = args.env_id.removeprefix("cocogrid/").removesuffix("-v0")
    if args.exp_name is None:
        run_name = f"rpo__{short_env}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.group_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            tags=args.wandb_tags.split(","),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    group_subdir = f"/{args.group_name}" if args.group_name is not None else ""
    if args.log_dir is None:
        log_base_dir = f"runs{group_subdir}"
    else:
        log_base_dir = args.log_dir
    log_dir = os.path.join(log_base_dir, run_name)
    if args.video_dir is None:
        video_base_dir = f"videos{group_subdir}"
    else:
        video_base_dir = args.video_dir
    video_dir = os.path.join(video_base_dir, run_name)

    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    next_eval_step = args.eval_interval
    next_checkpoint_step = args.checkpoint_freq

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(
        f"Using device {'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'}",
        device,
    )

    # env setup
    env_constructor = get_environment_constructor_for_id(args.env_id)
    env_kwargs = {}
    if is_cocogrid_env(args.env_id):
        env_kwargs = asdict(args.cocogrid)
        env_kwargs["timesteps"] = args.num_steps
        log_params = {
            "writer": writer,
            "max_steps": args.num_steps,
            "prefix": "cocogrid",
        }
    envs = gym.vector.SyncVectorEnv(
        [
            env_constructor(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                env_kwargs=env_kwargs,
                logging_params=log_params,
                video_dir=video_dir,
                goal_version=args.goal_version,
                reward_scale=args.reward_scale,
                norm_reward=args.reward_norm,
            )
            for i in range(args.num_envs)
        ]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [
            env_constructor(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                env_kwargs={**env_kwargs, "reset_options": {"eval": True}},
                logging_params={
                    "writer": writer,
                    "max_steps": args.num_steps,
                    "prefix": "validation",
                },
                video_dir=video_dir,
                goal_version=args.goal_version,
                reward_scale=args.reward_scale,
                norm_reward=args.reward_norm,
                is_eval=True,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = RPOAgent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        # evaluations
        if global_step >= next_eval_step:
            next_eval_step += args.eval_interval
            episode_count = 0
            step_count = 0
            LoggingWrapper.freeze_logging()  # clear out logs from unfinished episodes
            eval_obs, _ = eval_envs.reset()
            LoggingWrapper.resume_logging()
            for env in eval_envs.envs:
                env.enable_logging()
            eval_dones = np.zeros(args.num_envs)
            min_episodes_per_env = np.ceil(args.eval_episodes / args.num_envs)

            while (
                (dones < min_episodes_per_env).any()
                and step_count < args.num_steps * args.num_envs * min_episodes_per_env
            ):
                with torch.no_grad():
                    eval_action, _, _, _ = agent.get_action_and_value(
                        torch.Tensor(eval_obs).to(device)
                    )

                eval_obs, _, eval_terms, eval_truncs, eval_infos = eval_envs.step(
                    eval_action.cpu().numpy()
                )
                eval_dones = eval_dones + eval_terms + eval_truncs

                step_count += args.num_envs

                if "final_info" in eval_infos:
                    for idx, info in enumerate(eval_infos["final_info"]):
                        if (
                            info
                            and "episode" in info
                            and eval_dones[idx] < min_episodes_per_env
                        ):
                            print(
                                f"global_step={global_step}, envs_id={idx}, eval_count={episode_count}, episodic_return={info['episode']['r']}"
                            )
                            episode_count += 1
                            if eval_dones[idx] >= min_episodes_per_env:
                                eval_envs.envs[idx].disable_logging()

        # checkpoints
        if global_step >= next_checkpoint_step:
            torch.save(
                agent.state_dict(), f"{log_dir}/checkpoint_{next_checkpoint_step}.pth"
            )
            next_checkpoint_step += args.checkpoint_freq

    envs.close()
    eval_envs.close()
    writer.close()

    # checkpoints
    torch.save(agent.state_dict(), f"{log_dir}/final.pth")

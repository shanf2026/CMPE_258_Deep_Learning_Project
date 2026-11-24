"""SAC Atari training with optional action delay and observation frame drops.

This is a separate training script. Keep the clean baseline in sac_atari.py.
The script expects the normal CleanRL package layout and can be run from the
Code directory with:

    uv run python cleanrl/sac_atari_stress.py --env-id BreakoutNoFrameskip-v4 \
        --delay 1 --drop-prob 0.0 --capture-video
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.sac_atari import Actor, SoftQNetwork
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


class ActionDelayWrapper(gym.Wrapper):
    """Executes each action after a fixed number of decision steps."""

    def __init__(self, env, delay=0):
        super().__init__(env)
        if delay < 0:
            raise ValueError("delay must be nonnegative")
        self.delay = delay
        self.queue = deque(maxlen=delay + 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.queue.clear()
        for _ in range(self.delay):
            self.queue.append(0)  # Atari NOOP
        return obs, info

    def step(self, action):
        self.queue.append(action)
        return self.env.step(self.queue.popleft())


class FrameDropWrapper(gym.Wrapper):
    """With probability drop_prob, repeat the previous observation."""

    def __init__(self, env, drop_prob=0.0):
        super().__init__(env)
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be between 0 and 1")
        self.drop_prob = drop_prob
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.last_obs is not None and np.random.random() < self.drop_prob:
            returned_obs = self.last_obs
        else:
            self.last_obs = np.array(obs, copy=True)
            returned_obs = obs
        return returned_obs, reward, terminated, truncated, info


@dataclass
class Args:
    exp_name: str = "sac_atari_stress"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 5_000_000
    buffer_size: int = 200_000
    gamma: float = 0.99
    tau: float = 1.0
    batch_size: int = 64
    learning_starts: int = 20_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    update_frequency: int = 4
    target_network_frequency: int = 8_000
    alpha: float = 0.2
    autotune: bool = True
    target_entropy_scale: float = 0.89

    # Robustness conditions
    delay: int = 0
    drop_prob: float = 0.0

    # Keep each condition in its own directory.
    save_dir: str = "runs"


def make_env(args, seed, idx, run_name):
    def thunk():
        render_mode = "rgb_array" if args.capture_video and idx == 0 else None
        env = gym.make(args.env_id, render_mode=render_mode)
        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, os.path.join(args.save_dir, "videos", run_name)
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if args.delay > 0:
            env = ActionDelayWrapper(env, delay=args.delay)
        if args.drop_prob > 0.0:
            env = FrameDropWrapper(env, drop_prob=args.drop_prob)

        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk


def main():
    args = tyro.cli(Args)
    if args.delay < 0:
        raise ValueError("delay must be nonnegative")
    if not 0.0 <= args.drop_prob <= 1.0:
        raise ValueError("drop_prob must be between 0 and 1")

    condition = f"delay{args.delay}_drop{int(args.drop_prob * 100):02d}"
    run_name = f"{args.env_id}__{args.exp_name}__{condition}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.save_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args, args.seed, 0, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.single_action_space.n)
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.as_tensor(obs, device=device).float())
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts and global_step % args.update_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                _, next_log_pi, next_action_probs = actor.get_action(data.next_observations)
                qf1_next = qf1_target(data.next_observations)
                qf2_next = qf2_target(data.next_observations)
                min_q_next = next_action_probs * (
                    torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                )
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_q_next.sum(dim=1)

            qf1_values = qf1(data.observations)
            qf2_values = qf2(data.observations)
            qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
            qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
            qf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            _, log_pi, action_probs = actor.get_action(data.observations)
            with torch.no_grad():
                min_q_values = torch.min(qf1(data.observations), qf2(data.observations))
            actor_loss = (action_probs * ((alpha * log_pi) - min_q_values)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if args.autotune:
                alpha_loss = (
                    action_probs.detach()
                    * (-log_alpha.exp() * (log_pi + target_entropy).detach())
                ).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                sps = int(global_step / max(time.time() - start_time, 1e-6))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)

    final_path = os.path.join(run_dir, "sac_atari.pt")
    torch.save(actor.state_dict(), final_path)
    print(f"Saved model weights to {final_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

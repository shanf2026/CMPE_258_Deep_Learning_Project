# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
from collections import deque
import copy
import signal
import sys
from datetime import timedelta

import pickle
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # extra custom arguments for delay and frame drop
    delay : int = 0
    """how many frames to delay the input by"""
    drop_prob: float = 0.0
    """probability of a frame dropping"""

    # extra custom arguments for progress so I can resume later
    checkpoint_dir: str = "checkpoints"
    """directory to save periodic training checkpoints"""
    save_freq: int = 1000000
    """save checkpoint every N timesteps"""
    resume_from: str = ""
    """path to a .pt checkpoint file to resume training from"""


class ActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay=0):
        super().__init__(env)
        self.delay = delay
        self.queue = deque(maxlen=delay + 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.queue.clear()
        for _ in range(self.delay):
            self.queue.append(0)  # NOOP while initial inputs are in flight
        return obs, info

    def step(self, action):
        self.queue.append(action)
        return self.env.step(self.queue.popleft())

# i'm going to assume each obs array hasn't been modified
class FrameDropWrapper(gym.Wrapper):
    def __init__(self, env, drop_prob=0.0):
        super().__init__(env)
        self.drop_prob = drop_prob
        self.last_obs = None
        self.last_render = None
        self.dropped_this_step = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_render = None
        self.dropped_this_step = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.last_obs is not None and np.random.rand() < self.drop_prob:
            obs = self.last_obs
            self.dropped_this_step = True
        else:
            self.last_obs = obs
            self.dropped_this_step = False
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if self.dropped_this_step and self.last_render is not None:
            return self.last_render
        self.last_render = frame
        return frame


def make_env(env_id, seed, idx, capture_video, run_name, delay, drop_prob):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

        if delay > 0:
            env = ActionDelayWrapper(env, delay=delay)
        if drop_prob > 0.0:
            env = FrameDropWrapper(env, drop_prob=drop_prob)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# Capture SIGINT signal (Ctrl+C in Windows Powershell)
abort_flag = False

def handle_sigint(signum, frame):
    print(">> SIGINT CAPTURED! abort_flag set to True <<")
    global abort_flag
    abort_flag = True

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.delay, args.drop_prob) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    start_perf_counter = time.perf_counter()

    # custom section for resuming from an existing checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_step = 0

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device)
        q_network.load_state_dict(ckpt["q_network"])
        target_network.load_state_dict(ckpt["target_network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["global_step"] + 1

        # Optional: reload replay buffer if saved
        rb_path = args.resume_from + ".rb.pkl"
        if os.path.exists(rb_path):
            with open(rb_path, "rb") as f:
                rb = pickle.load(f)
            print("loaded checkpoint from", rb_path)
        print(f"Resumed at global_step={start_step}")



    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # modify to restart from start_step if --resume-from is used
    for global_step in range(start_step, args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    elapsed_steps = global_step - start_step
                    sps = elapsed_steps / (time.time() - start_time)
                    seconds = (args.total_timesteps - global_step) / max(sps, 1)
                    print("SPS:", int(sps), "ETA:", timedelta(seconds=int(seconds)))
                    writer.add_scalar("charts/SPS", int(elapsed_steps / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        if global_step <= args.learning_starts and abort_flag:
            print("shutting down immediately without saving due to still being in learning phase")
            envs.close()
            writer.close()
            sys.exit(0)

        # SAVING IN-PROGRESS STUFF FOR RESUMING LATER
        if global_step > args.learning_starts and (abort_flag or global_step % args.save_freq == 0):
            print("!!!!!!!! CHECKPOINT SAVING IN PROGRESS DO NOT TERMINATE !!!!!!!!!")
            start_perf_counter2 = time.perf_counter()
            run_name2 = run_name
            # TEMPORARY HALFWAY CHECKPOINT DELETE LATER
            if global_step == 5000000:
                run_name2 = run_name + "HALFWAY"
            ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name2}_latest.pt")
            torch.save(
                {
                    "global_step": global_step,
                    "q_network": q_network.state_dict(),
                    "target_network": target_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            # Optional: save replay buffer (~28 GB when full; comment out if saving weights/optimizer only)
            with open(ckpt_path + ".rb.pkl", "wb") as f:
                pickle.dump(rb, f)
            print(f"Saved checkpoint at step {global_step} -> {ckpt_path}")
            print(">>>>>>>>> SAVING COMPLETE <<<<<<<<")

            elapsed_perf_counter2 = time.perf_counter() - start_perf_counter2
            print(f"Checkpoint Saving Time: {timedelta(seconds=int(elapsed_perf_counter2))}")


            if abort_flag:
                print("Terminating early due to SIGINT")
                envs.close()
                writer.close()
                elapsed_perf_counter = time.perf_counter() - start_perf_counter
                print(f"Elapsed Time: {timedelta(seconds=int(elapsed_perf_counter))}")
                sys.exit(0)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        SKIP_EVALUATION = True

        if SKIP_EVALUATION:
            print("Final evaluation skipped!")
        else:
            from cleanrl_utils.evals.dqn_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                device=device,
                epsilon=args.end_e,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
    elapsed_perf_counter = time.perf_counter() - start_perf_counter
    print(f"Elapsed Time: {timedelta(seconds=int(elapsed_perf_counter))}")

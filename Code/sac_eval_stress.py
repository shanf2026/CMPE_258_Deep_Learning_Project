import argparse
import csv
import os
from collections import deque
import gymnasium as gym
import numpy as np
import torch
from cleanrl.sac_atari import Actor
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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


def make_eval_env(env_id, delay, drop_prob, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if delay > 0:
            env = ActionDelayWrapper(env, delay=delay)
        if drop_prob > 0.0:
            env = FrameDropWrapper(env, drop_prob=drop_prob)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .pt model")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--delay", type=int, default=0, help="Action delay in decision steps (1 step = 4 frames)")
    parser.add_argument("--drop-prob", type=float, default=0.0, help="Probability of freezing/dropping a frame (0.0 to 1.0)")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--csv-out", type=str, default="ppo_stress_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"eval_{args.env_id}_delay{args.delay}_drop{int(args.drop_prob * 100)}"
    envs = gym.vector.SyncVectorEnv(
        [make_eval_env(args.env_id, args.delay, args.drop_prob, args.capture_video, run_name)]
    )

    actor = Actor(envs).to(device)
    actor.load_state_dict(torch.load(args.model_path, map_location=device))
    actor.eval()

    obs, _ = envs.reset()
    returns = []
    lengths = []

    while len(returns) < args.episodes:
        with torch.no_grad():
            action, _, _, _ = actor.get_action(
                torch.as_tensor(obs, device=device).float()
            )
        obs, _, _, _, infos = envs.step(action.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    score = float(info["episode"]["r"])
                    ep_len = int(info["episode"]["l"])
                    returns.append(score)
                    lengths.append(ep_len)
                    print(f"Episode {len(returns)}/{args.episodes} | Score: {score} | Length: {ep_len}")

    mean_score = float(np.mean(returns))
    std_score = float(np.std(returns))
    mean_len = float(np.mean(lengths))
    print(f"\n=== Summary ({args.env_id} | delay={args.delay} | drop_prob={args.drop_prob}) ===")
    print(f"Mean Score: {mean_score:.2f} +/- {std_score:.2f} | Mean Survival Steps: {mean_len:.1f}")

    # Append summary row to CSV for easy plotting/tables later
    file_exists = os.path.isfile(args.csv_out)
    with open(args.csv_out, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["env_id", "model_path", "delay", "drop_prob", "episodes", "mean_score", "std_score", "mean_length"])
        writer.writerow([args.env_id, args.model_path, args.delay, args.drop_prob, args.episodes, round(mean_score, 2), round(std_score, 2), round(mean_len, 1)])

    envs.close()

# ppo_gridworld.py
import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment import GridWorldEnv

# ----------------------------- Utils -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def state_to_feat(env: GridWorldEnv, s: int) -> np.ndarray:
    r, c = env.state_tuple(s)
    N = env.N
    return np.array([r/(N-1), c/(N-1)], dtype=np.float32)

# ----------------------------- Model -----------------------------
class ActorCritic(nn.Module):
    """共享干路 + 两个头（actor/critic）。"""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.body(x)
        logits = self.actor(z)             # [B, A]
        value = self.critic(z).squeeze(-1) # [B]
        return logits, value

# ----------------------------- Config -----------------------------
@dataclass
class PPOConfig:
    N: int = 5
    seed: int = 2025
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    total_steps: int = 50_000          # 训练总步数（交互步）
    rollout_steps: int = 2048          # 每轮收集这么多步
    epochs: int = 10                   # 每轮优化的 epoch 次数
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2              # PPO clip
    vf_clip_eps: float = 0.2           # value clipping（可选）
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_ep_len: int = 200              # 每个 episode 步长上限
    print_every_steps: int = 4096
    eval_episodes: int = 20

# ----------------------------- Buffer -----------------------------
class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.ptr += 1

    def full(self):
        return self.ptr >= self.capacity

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        adv = 0.0
        for t in reversed(range(self.ptr)):
            mask = 1.0 - self.dones[t]
            next_value = last_value if t == self.ptr - 1 else self.values[t+1]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_minibatches(self, batch_size: int):
        idx = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            j = idx[start:start+batch_size]
            yield (self.obs[j], self.actions[j], self.logprobs[j],
                   self.returns[j], self.advantages[j], self.values[j])

    def reset(self):
        self.ptr = 0

# ----------------------------- PPO Trainer -----------------------------
class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.env = GridWorldEnv(N=cfg.N, seed=cfg.seed)
        self.obs_dim = 2
        self.act_dim = self.env.A
        self.net = ActorCritic(self.obs_dim, self.act_dim, hidden=128).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def select_action(self, s: int):
        feat = torch.tensor(state_to_feat(self.env, s), device=self.device).unsqueeze(0)
        logits, value = self.net(feat)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(value.item())

    def train(self):
        cfg = self.cfg
        device = self.device
        buffer = RolloutBuffer(cfg.rollout_steps, self.obs_dim, device)
        s = self.env.reset()
        global_steps = 0
        ep_return, ep_len = 0.0, 0
        smoothed = None

        while global_steps < cfg.total_steps:
            # ----------- collect rollout -----------
            buffer.reset()
            while not buffer.full():
                obs = torch.tensor(state_to_feat(self.env, s), device=device)
                with torch.no_grad():
                    logits, v = self.net(obs.unsqueeze(0))
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    logp = dist.log_prob(a)

                ns, r, done = self.env.step(int(a.item()))
                buffer.add(obs, a.item(), logp.item(), r, done, v.item())

                s = ns
                ep_return += r; ep_len += 1
                global_steps += 1

                if done or ep_len >= cfg.max_ep_len:
                    s = self.env.reset()
                    smoothed = ep_return if smoothed is None else 0.9 * smoothed + 0.1 * ep_return
                    ep_return, ep_len = 0.0, 0

            # ----------- GAE -----------
            with torch.no_grad():
                last_feat = torch.tensor(state_to_feat(self.env, s), device=device).unsqueeze(0)
                _, last_v = self.net(last_feat)
                last_v = float(last_v.item())
            buffer.compute_gae(last_value=last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)

            # 标准化优势
            adv = buffer.advantages[:buffer.ptr]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            buffer.advantages[:buffer.ptr] = adv

            # ----------- PPO updates -----------
            for _ in range(cfg.epochs):
                for (obs_b, act_b, old_logp_b, ret_b, adv_b, old_v_b) in buffer.get_minibatches(cfg.minibatch_size):
                    logits, v_pred = self.net(obs_b)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(act_b)
                    entropy = dist.entropy().mean()

                    # ratio
                    ratio = torch.exp(logp - old_logp_b)

                    # policy loss (clip)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss (with optional clip)
                    # v_clip = old_v + clamp(v - old_v, -eps, +eps)
                    v_clipped = old_v_b + torch.clamp(v_pred - old_v_b, -cfg.vf_clip_eps, cfg.vf_clip_eps)
                    v_loss1 = (v_pred - ret_b).pow(2)
                    v_loss2 = (v_clipped - ret_b).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                    self.opt.step()

            if global_steps % cfg.print_every_steps < cfg.rollout_steps:
                print(f"steps={global_steps:6d} | smoothed_return={smoothed if smoothed is not None else 0.0:8.2f}")

        print("Training done.")

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float]:
        cfg = self.cfg
        total_r, total_len, succ = 0.0, 0, 0
        for _ in range(cfg.eval_episodes):
            s = self.env.reset()
            ep_r, steps = 0.0, 0
            for t in range(cfg.max_ep_len):
                feat = torch.tensor(state_to_feat(self.env, s), device=self.device).unsqueeze(0)
                logits, _ = self.net(feat)
                a = torch.argmax(logits, dim=-1).item()
                s, r, done = self.env.step(a)
                ep_r += r; steps += 1
                if done:
                    succ += 1
                    break
            total_r += ep_r
            total_len += steps
        return total_r / cfg.eval_episodes, total_len / cfg.eval_episodes, succ / cfg.eval_episodes

# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    cfg = PPOConfig(
        N=5,
        total_steps=60_000,
        rollout_steps=2048,
        epochs=8,
        minibatch_size=256,
        clip_eps=0.2,
        vf_clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_ep_len=200,
        lr=3e-4,
        seed=2025,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    agent = PPOAgent(cfg)
    agent.train()
    avg_r, avg_len, succ = agent.evaluate()
    print(f"[Eval] avg_return={avg_r:.2f} | avg_len={avg_len:.1f} | success_rate={succ*100:.1f}%")

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import SmoothL1Loss
import torch.optim as optim
from torch.distributions import Categorical

# ---- import your env ----
from environment import GridWorldEnv

def set_seed(seed: int):
    """set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def state_to_feat(env: GridWorldEnv, s: int):
    """Simple features: normalized row/col plus goal as constant hint."""
    r, c = env.state_tuple(s)
    N = env.N
    # normalize to [0,1]
    return np.array([r/(N-1), c/(N-1)], dtype=np.float32)

# ---------------- models ----------------


class PolicyNet(nn.Module):
    """Model whose input is the state,giving the probabilities of the action, which is the actor"""
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits
        )
    def forward(self, x):  # x: [B, in_dim]
        return self.net(x)  # logits

class ValueNet(nn.Module):
    """Model whose input is the state, giving the socre of that state, which is the critic"""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

# ---------------- config ----------------
@dataclass
class Config:
    N: int = 5
    seed: int = 2025
    device: str = "cuda"
    hidden: int = 128
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_steps_per_episode: int = 100
    episodes: int = 500
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    print_every: int = 100
    eval_every: int = 200
    eval_episodes: int = 100
    Smoothed_return: bool = True

# ---------------- training ----------------
def train_actor_critic(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = GridWorldEnv(N=cfg.N, seed=cfg.seed)
    obs_dim = 2
    act_dim = env.A

    actor = PolicyNet(obs_dim, cfg.hidden, act_dim).to(device)
    critic = ValueNet(obs_dim, cfg.hidden).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    smoothed_return = cfg.Smoothed_return

    for ep in range(1, cfg.episodes + 1):
        s = env.reset()
        ep_return = 0.0
        for t in range(cfg.max_steps_per_episode):
            # ---- prepare tensors ----
            feat = torch.tensor(state_to_feat(env, s), device=device).unsqueeze(0)  # [1,2]
            logits = actor(feat)                                    # [1,4]
            dist = Categorical(logits=logits)
            a = dist.sample()                                       # [1]
            logp = dist.log_prob(a)                                 # [1]
            ent = dist.entropy()                                    # [1]
                                                                    # add entropy loss to inspire exploration
            s_next, r, done = env.step(int(a.item()))
            ep_return += r

            # ---- critic targets and TD error (Advantage) ----
            with torch.no_grad():
                v_next = 0.0 if done else critic(torch.tensor(state_to_feat(env, s_next), device=device).unsqueeze(0)).item()
                target = r + cfg.gamma * v_next
            v = critic(feat)                                        # [1]
            td_error = target - v                                   # [1]

            # ---- losses ----
            policy_loss = -(logp * td_error.detach()) - cfg.entropy_coef * ent
            value_loss = 0.5 * td_error.pow(2)

            # ---- optimize ----
            opt_actor.zero_grad()
            policy_loss.mean().backward()
            opt_actor.step()

            opt_critic.zero_grad()
            value_loss.mean().backward()
            opt_critic.step()

            s = s_next
            if done:
                break

        smoothed_return = ep_return if smoothed_return is None else 0.9*smoothed_return + 0.1*ep_return
        if ep % cfg.print_every == 1 or ep == cfg.episodes:
            print(f"Ep {ep:4d} | steps={t+1:3d} | return={ep_return:8.2f} | smoothed={smoothed_return:8.2f}")

        if ep % cfg.eval_every == 0:
            avg_r, avg_len, succ = evaluate_policy(env, actor, cfg)
            print(f"  [Eval] avg_return={avg_r:.2f} | avg_len={avg_len:.1f} | success_rate={succ*100:.1f}%")

    return actor, critic, env

# ---------------- evaluation ----------------
@torch.no_grad()
def evaluate_policy(env: GridWorldEnv, actor: PolicyNet, cfg: Config):
    device = torch.device(cfg.device)
    total_r, total_len, succ = 0.0, 0, 0
    for _ in range(cfg.eval_episodes):
        s = env.reset()
        ep_r, steps = 0.0, 0
        for t in range(cfg.max_steps_per_episode):
            feat = torch.tensor(state_to_feat(env, s), device=device).unsqueeze(0)
            logits = actor(feat)
            # greedy eval
            a = torch.argmax(logits, dim=-1).item()
            s, r, done = env.step(a)
            ep_r += r
            steps += 1
            if done:
                succ += 1
                break
        total_r += ep_r
        total_len += steps
    return total_r / cfg.eval_episodes, total_len / cfg.eval_episodes, succ / cfg.eval_episodes

# ---------------- run ----------------
if __name__ == "__main__":
    cfg = Config(N=5, device="cuda", episodes=500, entropy_coef=0.02, max_steps_per_episode=100,Smoothed_return=True)
    actor, critic, env = train_actor_critic(cfg)
    avg_r, avg_len, succ = evaluate_policy(env, actor, cfg)
    print(f"\nFinal Eval → avg_return={avg_r:.2f}, avg_len={avg_len:.1f}, success_rate={succ*100:.1f}%")

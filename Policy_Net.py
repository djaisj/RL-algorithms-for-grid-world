from environment import GridWorldEnv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def _softmax(logits: torch.Tensor, dim: int =-1) -> torch.Tensor:
    logits = torch.tensor(logits, dtype=torch.float32) if not isinstance(logits, torch.Tensor) else logits
    z = logits - logits.max(dim=dim, keepdim=True).values
    ez = torch.exp(z)
    return ez / ez.sum(dim=dim, keepdim=True)

def _state_features(env,s):
    """Map state id -> normalized (r, c) ∈ [0,1]^2 as NN input."""
    r,c = env.state_tuple(s)
    return torch.tensor([r/env.N,c/env.N],dtype = torch.float32)

class PolicyNet(nn.Module):
    def __init__(self,input_dim=2,hidden = 64,num_actions = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
    def forward(self,x):
        return self.net(x)

def _discounted_return(rewards,gamma):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

@torch.no_grad()
def evaluate_policy_net(env,policy_net,episodes=200,max_steps= 1000,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.eval()
    total = 0.0

    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            feat = _state_features(env,s).unsqueeze(0).to(device)
            logits = policy_net(feat)
            probs = _softmax(logits.cpu().numpy().squeeze(0))
            probs = probs.numpy() if isinstance(probs, torch.Tensor) else probs
            a = int(np.random.choice(env.A,p=probs))
            ns,r,done = env.step(a)
            ep_ret += r
            steps += 1
            s = ns
        total += ep_ret
    return total / episodes

def train_policy_net(
        env,
        episodes=1500,
        gamma=0.99,
        lr=1e-2,
        max_steps_per_episode=200,
        entropy_coef=1e-2,
        grad_clip=0.5,
        normalize_returns=True,
        seed = 0,
        log_every = 100,
        device = None
):
    np.random.seed(seed); torch.manual_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNet(input_dim=2, hidden=64, num_actions=env.A).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for ep in range(episodes):
        policy_net.train()
        logps,entropies,rewards = [],[],[]

        s = env.reset()
        done = False

        steps = 0
        while not done and steps <max_steps_per_episode:
            feat = _state_features(env,s).unsqueeze(0).to(device)
            logits = policy_net(feat)
            probs = _softmax(logits.squeeze(0))
            a = torch.multinomial(probs, num_samples=1)              # [1,1]                 # [1]
            a_int = int(a.item())
            logp = torch.log(probs[a_int])
                            # 纯标量 (0-dim tensor)
            ns,r,done = env.step(a_int)
            ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1)#在训练里通常乘上 entropy_coef 加进 loss，鼓励策略保持探索，避免过早收敛。
            logps.append(logp)
            entropies.append(ent)
            rewards.append(r)
            s = ns
            steps += 1

        returns = _discounted_return(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        logps = torch.stack(logps)           # [T]
        entropies = torch.stack(entropies)   # [T]

        # REINFORCE：maximize E[G * log π] -> minimize -(G * log π) yse it is right, just using the integral to process the original equation we will get tht maximize object
        pg_loss = -(logps * returns).sum()
        entropy_loss = -entropy_coef * entropies.sum() # make the probability more flat
        loss = pg_loss + entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad.clip_grad_norm_(policy_net.parameters(), grad_clip)
        optimizer.step()

        if (ep + 1) % log_every == 0:
            reached = (len(rewards) < max_steps_per_episode and len(rewards) > 0 and rewards[-1] == env.r_goal)
            print(f"[ep {ep+1}/{episodes}] steps={len(rewards):4d} reached_goal={bool(reached)} loss={loss.item():.3f}")

    return policy_net


if __name__ == "__main__":
    env = GridWorldEnv(5,0);
    Net = PolicyNet()
    np.random.seed(0)
    env = GridWorldEnv(N=5, seed=0)
    Net = PolicyNet(input_dim=2,hidden = 2,num_actions = 4)
    Net = train_policy_net(
        env,
        episodes=200,
        gamma=0.99,
        lr=1e-2,
        max_steps_per_episode=200,
        entropy_coef=1e-2,
        grad_clip=0.5,
        normalize_returns=True,
        seed = 0,
        log_every = 10,
        device = None)
    avg_reward = evaluate_policy_net(env,Net,episodes=200,max_steps= 1000,device=None)
    print(avg_reward)

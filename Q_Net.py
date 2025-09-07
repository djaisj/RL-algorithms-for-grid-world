import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import math

def _state_features(env,s):
    """Map state id -> normalized (r, c) ∈ [0,1]^2 as NN input."""
    r,c = env.state_tuple(s)
    return torch.tensor([r/env.N,c/env.N],dtype = torch.float32)

class QNET(nn.Module): # 会得到action,也是跟q table 一样的
    def __init__(self,input_dim= 2,hidden = 64,num_actions = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
    def forward(self,x):
        return self.net(x)


Transition = namedtuple('Transition', ('s', 'a', 'r', 'ns', 'done'))

def dqn_learning(
  env,
  episodes=800,
  gamma=0.99,
  lr=3e-3,
  batch_size=64,
  buffer_size = 5000,
  epsilon_start=1.0,
  epsilon_end=0.05,
  epislon__decay = 400,
  target_sync = 200,
  hidden = 64,
  seed = None,
  device = None,
  q_net = None

):
    """
    Train a DQN on the given env. Returns (q_net, info_dict).
    Uses (r,c) normalized coords as features -> real function approximation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    if q_net is None:
        q_net = QNET(input_dim=2, hidden=hidden, num_actions=env.A).to(device)
    if next(q_net.parameters()).device != device:
        q_net = q_net.to(device)
    tgt_net = QNET(input_dim=2, hidden=hidden, num_actions=env.A).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    tgt_net.eval() # make sure the target network is not trainable

    # the target stays fixed for a while, so the learner can “catch up”.

    # Then we update the target to the new parameters.

    # This reduces oscillation and divergence.
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss() # Huber loss 

    replay = deque(maxlen=buffer_size)

    def epislon_by_step(step):
        return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step / epislon__decay)  #This function is just a schedule to gradually reduce randomness as the agent learns, moving from full exploration toward mostly exploitation of the learned policy.
    global_step = 0
    ep_returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            eps = epislon_by_step(global_step)
            if np.random.rand() < eps:
                a = np.random.randint(env.A)
            else:
                with torch.no_grad():
                    feat = _state_features(env,s).unsqueeze(0).to(device)
                    qvals = q_net(feat)
                    a = int(torch.argmax(qvals,dim=1).item())
            ns,r, done = env.step(a)
            replay.append(Transition(s,a,r,ns,float(done)))
            ep_ret += r
            s = ns
            global_step += 1

            if len(replay) >= batch_size:
                batch = [replay[idx] for idx in np.random.randint(len(replay), size=batch_size)]
                bs = torch.stack([_state_features(env, t.s) for t in batch]).to(device)        # [B,2]
                bns = torch.stack([_state_features(env, t.ns) for t in batch]).to(device)      # [B,2]
                ba = torch.tensor([t.a for t in batch], dtype=torch.long, device=device)       # [B]
                br = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device)    # [B]
                bdone = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                q_sa = q_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next = tgt_net(bns).max(dim=1).values
                    target = br + (1.0-bdone) * gamma * max_next
                loss = loss_fn(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)  # 保证梯度不爆炸
                optimizer.step() 

            if global_step % target_sync == 0:
                tgt_net.load_state_dict(q_net.state_dict()) 

        ep_returns.append(ep_ret)
    
    info = {
        'episode_returns': np.array(ep_returns,dtype = float),
        'steps':global_step
    }                      
    
    return q_net,info 

def evaluate_policy_net(env,q_net,episodes=100,device = None,max_steps = 200):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              # 用梯度更新参数
    q_net.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            s = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            while not done and steps < max_steps:
                feat = _state_features(env,s).unsqueeze(0).to(device)
                qvals = q_net(feat)
                a = int(torch.argmax(qvals,dim=1).item())
                s,r,done = env.step(a)
                steps += 1
                ep_ret += r
            total += ep_ret
    return total / episodes


if __name__ == "__main__":
    from environment import GridWorldEnv
    import matplotlib.pyplot as plt

    np.random.seed(0)

    env = GridWorldEnv(N=5, seed=0)

    reward = []

    QNet,info = dqn_learning(
  env,
  episodes=401,
  gamma=0.99,
  lr=3e-4,
  batch_size=64,
  buffer_size = 100,
  epsilon_start=1.0,
  epsilon_end=0.05,
  epislon__decay = 400,
  target_sync = 50,
  hidden = 16,
  seed = None,
  device = None
)
    avg_reward = evaluate_policy_net(env,QNet,episodes=100,device = None,max_steps = 200)
    print(avg_reward)
    reward.append(avg_reward)
    for _ in range(1):
        QNet,info = dqn_learning(
                                env,
                                episodes=401,
                                gamma=0.99,
                                lr=3e-4,
                                batch_size=64,
                                buffer_size = 100,
                                epsilon_start=1.0,
                                epsilon_end=0.05,
                                epislon__decay = 400,
                                target_sync = 50,
                                hidden = 16,
                                seed = None,
                                device = None,
                                q_net = QNet
                                )
        avg_reward = evaluate_policy_net(env,QNet,episodes=100,device = None,max_steps=200)
        print(avg_reward)
        reward.append(avg_reward)

    plt.plot(range(len(reward)), reward, marker='o')
    plt.xlabel("Training block (21 episodes each)")
    plt.ylabel("Avg return over 100 eval episodes")
    plt.show()


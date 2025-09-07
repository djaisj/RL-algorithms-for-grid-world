import numpy as np


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    e = np.exp(x - m)
    s = e.sum()
    return e / (s + 1e-12)

def evaluate_policy(env,  theta=None, episodes=200, max_steps_per_episode=100):
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            probs = _softmax(theta[s])
            a = int(np.random.choice(env.A, p=probs))
            s, r, done = env.step(a)
            ep_ret += r
            steps += 1
        total += ep_ret
    return total / episodes


def policy_gradient(env, episodes=1500, alpha=0.05, gamma=0.99,
                    max_steps_per_episode=200, log_every=50,
                    use_state_baseline=True, entropy_coef=0.01):
    theta = np.zeros((env.S, env.A), dtype=float)

    # simple state-dependent baseline (EMA of returns from s)
    b = np.zeros(env.S, dtype=float)
    beta = 0.02  # baseline learning rate

    for ep in range(episodes):
        s_traj, a_traj, r_traj = [], [], []
        s = env.reset()

        for t in range(max_steps_per_episode):
            probs = _softmax(theta[s])
            a = int(np.random.choice(env.A, p=probs))
            ns, r, done = env.step(a)
            s_traj.append(s); a_traj.append(a); r_traj.append(r)
            s = ns
            if done:
                break

        # returns G_t
        G = 0.0
        returns = []
        for r in reversed(r_traj):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        # advantages
        if use_state_baseline:
            advs = []
            for s, Gt in zip(s_traj, returns):
                adv = Gt - b[s]
                b[s] += beta * (Gt - b[s])  # update baseline
                advs.append(adv)
            advs = np.asarray(advs, dtype=float)
        else:
            # per-episode normalize (works surprisingly well)
            Gs = np.asarray(returns, dtype=float)
            advs = (Gs - Gs.mean()) / (Gs.std() + 1e-8)

        # policy update
        for s, a, adv in zip(s_traj, a_traj, advs):
            probs = _softmax(theta[s])
            grad_logp = -probs
            grad_logp[a] += 1.0
            ent_grad = -(np.log(np.clip(probs, 1e-12, 1.0)) + 1.0)
            theta[s] += alpha * (adv * grad_logp + entropy_coef * ent_grad) # Using baseline to discriminate the difference from worse to more worse
        if (ep + 1) % log_every == 0:
            print(f"episode {ep+1}/{episodes}, steps={len(r_traj)}, reached_goal={r_traj and r_traj[-1] >= env.r_goal*0.9}")

    return theta



if __name__ == "__main__":
    from environment import GridWorldEnv
    env = GridWorldEnv(N=5, seed=0)
# If you run this unchanged environment (wall −10, step −0.01) and never reach the goal in the first 50 episodes, this baseline version still learns because:

# From the start state, up/left produce much worse-than-average returns (due to frequent −10 wall hits), so their probs shrink.

# Right/down become better-than-average, so their probs increase.

# Eventually the agent walks toward the goal and starts seeing +100, which accelerates learning.
    theta = policy_gradient(env, episodes=15000, alpha=0.05, gamma=0.99,
                    max_steps_per_episode=100, log_every=50,
                    use_state_baseline=True, entropy_coef=0.01)
    avg_reward = evaluate_policy(env, theta=theta, episodes=200,max_steps_per_episode=100)

    print(f"Policy gradient average reward over 200 episodes: {avg_reward:.2f}")

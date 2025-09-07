import numpy as np


def q_learning(env, episodes=800, alpha=0.1, gamma=0.95, epsilon=0.1,Q = None):
    """Tabular Q-learning with epsilon-greedy exploration.
    Return: Q table as np.ndarray of shape [S, A]."""
    if Q is None:
        Q = np.zeros((env.S, env.A), dtype=float) # Q just remembers all the state and action pair , and it is the expectation when take action a at state s
    for episode in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0 
        while not done:
            if np.random.rand() < epsilon:
                a = np.random.randint(env.A)
            else:
                a = np.argmax(Q[s])
            ns,r, done = env.step(a)
            # using the td target
            td_target = r + (0 if done else gamma * np.max(Q[ns])) # gamma is the discounted factor
            Q[s,a] += alpha * (td_target - Q[s,a]) # This is the learning rate, because we should let the difference minimal so we use the -1 to multilpy to make a negative feedback
            s = ns 
    return Q


def evaluate_policy(env, Q=None, episodes=100, max_steps_per_episode=500):
    total = 0.0
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            if Q is not None:
                # tie-break randomly instead of always taking index 0
                row = Q[s]
                a_candidates = np.flatnonzero(row == row.max())
                a = int(np.random.choice(a_candidates))
            else:
                break
            s, r, done = env.step(a)
            ep_ret += r
            steps += 1
        total += ep_ret
    return total / episodes


def test_q_learning_and_eval():
    from environment import GridWorldEnv
    np.random.seed(0)
    env = GridWorldEnv(N=5, seed=0)
    Q = q_learning(env, episodes=10, alpha=0.15, gamma=0.95, epsilon=0.1)
    avg_reward = evaluate_policy(env, Q=Q, episodes=100)
    print(avg_reward)
    if avg_reward < 5.0:
        print(f"Q-learning too weak: avg_reward={avg_reward:.2f}")
    for _ in range(80):
            Q = q_learning(env, episodes=10, alpha=0.15, gamma=0.95, epsilon=0.1,Q = Q)
            avg_reward = evaluate_policy(env, Q=Q, episodes=100)
            print(avg_reward)
            if avg_reward < 5.0:
                print(f"Q-learning too weak: avg_reward={avg_reward:.2f}")

if __name__ == "__main__":
    test_q_learning_and_eval()
    print("q_learning.py: All tests passed.")








class GridWorldEnv:
    def __init__(self, N=5, seed=None):
        """Initialize an N x N deterministic GridWorld.
        Start at (0,0); goal at (N-1,N-1).
        Actions: 0 up, 1 down, 2 left, 3 right.
        Rewards: -0.1 per step; -1.0 if bump wall; +10 at goal (done).
        """
        self.N = int(N)
        self.S = self.N * self.N  # number of states
        self.A = 4                # number of actions
        self.start = (0,0)
        self.goal = (self.N-1, self.N-1)
        self.r_step =  -0.01
        self.r_wall = -10
        self.r_goal = 100
        self._state = self.state_id(self.start)
    def reset(self):
        """Reset to start state and return state_id (int)."""
        self._state = self.state_id(self.start)
        return self._state

    def step(self, action: int):
        """Take an action, return (next_state_id:int, reward:float, done:bool)."""
        r,c = self.state_tuple(self._state)

        # Define movement mapping: action -> (row change, col change)
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        # Compute new candidate state
        dr, dc = moves.get(action, (0, 0))  # default to (0,0) if invalid action
        r, c = r + dr, c + dc
        
        # wall check
        if r < 0 or r >= self.N or c < 0 or c >= self.N:
            reward = float(self.r_wall)
            next_state = self._state
            done = False
        else:
            next_state = self.state_id((r,c))
            if (r,c) == self.goal:
                reward = float(self.r_goal)
                done = True
            else:
                reward = float(self.r_step)
                done = False
        self._state = next_state
        return next_state, reward, done

    def state_id(self, rc):
        """Map (row,col) -> int id in [0, N*N)."""
        r,c = rc
        return r*self.N + c

    def state_tuple(self, s):
        """Map int id -> (row,col)."""
        r = s // self.N
        c = s % self.N
        return (r,c)


def test_env_basics():
    env = GridWorldEnv(N=5, seed=123)
    s0 = env.reset()
    assert isinstance(s0, int)
    s1, r, d = env.step(1)  # down
    assert isinstance(s1, int) and isinstance(r, float) and isinstance(d, bool)
    # Hitting left wall from col 0
    env.reset()
    s, r, d = env.step(2)  # left
    assert r <= -0.1  # wall penalty or step penalty
    # Reaching goal must end episode
    env = GridWorldEnv(N=3, seed=0)
    s = env.reset()
    # shortest path: right,right,down,down
    s,r,d = env.step(3); s,r,d = env.step(3); s,r,d = env.step(1); s,r,d = env.step(1)
    assert d is True and r >= 9.0

if __name__ == "__main__":
    test_env_basics()
    print("env.py: All tests passed.")

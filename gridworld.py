import numpy as np

# S = 5*5 table, 25 grid positions, (0,0) start, (4,4) goal
# A = 0 = up, 1 = right, 2 = down, 3 = left
# R = -1 for each step, 0 for the goal
# P = transitions are deterministic,
# given a state and action, there is one resulting next state.
# Attempting to cross a boundary results in no movement.
# γ = used by agent later

class GridWorld:
    def __init__(self):
        self.size = 5
        self.start = (0,0)
        self.goal = (4,4)
        self.position = self.start

        self.n_states = self.size * self.size
        self.n_actions = 4

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        row, col = self.position
        if action == 0:  # up
            if row > 0:
                row -= 1
        elif action == 1:  # right
            if col < self.size - 1:
                col += 1
        elif action == 2:  # down
            if row < self.size - 1:
                row += 1
        elif action == 3:  # left
            if col > 0:
                col -= 1
        else:
            raise ValueError("Action must be 0, 1, 2, or 3")

        self.position = (row, col)
        if(self.position != self.goal):
            reward = -1
            done = False
        else:
            reward = 0
            done = True

        return self.position, reward, done

    def convert(self, position):
        row, col = position
        state = row * self.size + col
        return state

def run_random_episode(env):       # Seperate agent and environment class
        rng = np.random.default_rng()
        env.reset()     # Reset environment for next episode
        total_reward = 0
        total_steps = 0

        while(True):
            action = rng.integers(env.n_actions)
            position, reward, done = env.step(action)
            total_reward += reward
            total_steps += 1

            if(done == True):
                break

        return total_reward, total_steps





if __name__ == "__main__":
    env1 = GridWorld()      # Testing
    print(run_random_episode(env1))
    print(run_random_episode(env1))
    print(run_random_episode(env1))
    print(run_random_episode(env1))
    print(run_random_episode(env1))



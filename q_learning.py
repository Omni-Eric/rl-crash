import numpy as np
from gridworld import GridWorld

env = GridWorld()       #initialize environment and Q table
Q = np.zeros((env.n_states, env.n_actions))
print(Q.shape)

def epsilon_greedy(Q, state, epsilon, rng):
    r_number1 = rng.random()
    if(r_number1 < epsilon):
        r_index = rng.integers(Q.shape[1])
        return r_index
    else:
        return np.argmax(Q[state])

env.reset()     # Set up before training
state = env.convert(env.position)
done = False
total_reward = 0
steps = 0
rng = np.random.default_rng()
alpha = 0.3
gamma = 0.8

while(not done):        # One training episode
    action = epsilon_greedy(Q, state, 0.1, rng)
    next_position, reward, done = env.step(action)
    next_state = env.convert(next_position)
    old_q = Q[state, action]

    if done:
        target = reward
    else:
        best_next_q = np.max(Q[next_state])
        target = reward + gamma * (best_next_q)

    td_error = target - old_q
    Q[state, action] = old_q + alpha * td_error # update the q table

    total_reward += reward
    steps += 1

    print(
    "state:", state,
    "action:", action,
    "reward:", reward,
    "next_state:", next_state,
    "old Q:", old_q,
    "target:", target,
    "new Q:", Q[state, action]
)

    state = next_state # Move to next state

print("Total reward:", total_reward)
print("Steps:", steps)
    
    















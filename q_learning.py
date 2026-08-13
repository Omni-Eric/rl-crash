import numpy as np
from gridworld import GridWorld

env = GridWorld()       #initialize environment

def epsilon_greedy(Q, state, epsilon, rng):
    r_number1 = rng.random()
    if(r_number1 < epsilon):
        r_index = rng.integers(Q.shape[1])
        return r_index
    else:
        return np.argmax(Q[state])

def train_q_learning(
    env,
    episodes,
    alpha,
    gamma,
    epsilon,
    seed
):
    Q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(seed)
    episode_returns = []
    episode_steps = []

    for i in range(episodes):
        position = env.reset()     # Set up before training
        state = env.convert(position)
        done = False
        total_reward = 0
        steps = 0

        while(not done):        # One training episode
            action = epsilon_greedy(Q, state, epsilon, rng)
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

            state = next_state # Move to next state

        episode_returns.append(total_reward)
        episode_steps.append(steps)
    return Q, episode_returns, episode_steps

   
Q, returns, steps = train_q_learning(env, 500, 0.1, 0.9, 0.1, 0)
print(returns)
print(steps)
print(len(returns))
print(len(steps))





    
    















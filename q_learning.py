import matplotlib.pyplot as plt
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

def evaluate(Q, env):
    position = env.reset()     
    state = env.convert(position)
    done = False
    steps = 0
    path = [position]

    while(not done):
        action = np.argmax(Q[state])
        next_position, reward, done = env.step(action)
        path.append(next_position)
        next_state = env.convert(next_position)

        steps += 1
        state = next_state

    return steps, path

def plot_learning_curve(returns):
    episode_numbers = np.arange(1, len(returns) + 1)
    plt.plot(episode_numbers, returns)               # draw the data
    plt.xlabel("Episode")       # x-axis name
    plt.ylabel("Episode Return")        # y-axis name
    plt.title("Learning Curve with Episodes")            # graph title
    plt.show()                  

def moving_average(returns, window=20):
    weights = np.ones(window) / window

    smoothed_returns = np.convolve(
    returns,
    weights,
    mode="valid"
)
    return smoothed_returns

# Experiment A Baseline
Q_A, returns_A, steps_A = train_q_learning(env, 500, 0.1, 0.9, 0.1, 0)      
# Experiment B 
Q_B, returns_B, steps_B = train_q_learning(env, 500, 0.1, 0.9, 0.3, 0)      # Higher Exploration
# Experiment C 
Q_C, returns_C, steps_C = train_q_learning(env, 500, 0.5, 0.9, 0.1, 0)      # Higher learning rate

smooth_A = moving_average(returns_A, 20)
smooth_B = moving_average(returns_B, 20)
smooth_C = moving_average(returns_C, 20)

episodes = np.arange(20, 501)

plt.plot(episodes, smooth_A, label="Baseline: α=0.1, ε=0.1")
plt.plot(episodes, smooth_B, label="More exploration: α=0.1, ε=0.3")
plt.plot(episodes, smooth_C, label="Higher learning rate: α=0.5, ε=0.1")

plt.xlabel("Episode")
plt.ylabel("Moving Average Episode Return")
plt.title("Q-Learning on 5x5 GridWorld")
plt.legend()
plt.savefig(
    "figures/q_learning_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


a, b = evaluate(Q_A, env)     
print("Learning path:", b, "\n", "Steps:", a)

a, b = evaluate(Q_B, env)     
print("Learning path:", b, "\n", "Steps:", a)

a, b = evaluate(Q_C, env)     
print("Learning path:", b, "\n", "Steps:", a)





# plot_learning_curve(returns)        # Plot learning curve












    
    















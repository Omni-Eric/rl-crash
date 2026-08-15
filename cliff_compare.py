import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v0")

def epsilon_greedy(Q, state, epsilon, rng):     # Same for both Q-learning and SARSA
    r_number = rng.random()

    if r_number < epsilon:
        return rng.integers(Q.shape[1])
    else:
        return np.argmax(Q[state])

def train_q_learning(       # Training Loop for Q-learning
    env,
    episodes,
    alpha,
    gamma,
    epsilon,
    seed
):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(seed)
    episode_returns = []
    episode_steps = []

    for i in range(episodes):
        state, info = env.reset()     # Set up before training
        done = False
        total_reward = 0
        steps = 0

        while(not done):        # One training episode
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            old_q = Q[state, action]

            if terminated:
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

def train_sarsa(        # Training Loop for SARSA
    env,
    episodes,
    alpha,
    gamma,
    epsilon,
    seed
):

    rng = np.random.default_rng(seed)
    Q = np.zeros(
    (env.observation_space.n, env.action_space.n)
)
    episode_returns = []
    episode_steps = []

    for i in range(episodes):
        state, info = env.reset()

        
        # SARSA needs an action BEFORE entering the loop.
        action = epsilon_greedy(Q, state, epsilon,rng)

        done = False
        total_reward = 0
        steps = 0

        while not done:

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            old_q = Q[state, action]

            if terminated:
                target = reward
            else:
                
                # Choose the ACTUAL next action using epsilon-greedy.
                next_action = epsilon_greedy(Q, next_state, epsilon,rng)

                
                # SARSA uses Q[next_state, next_action]
                target = reward + gamma * Q[next_state, next_action]

            td_error = target - old_q

            Q[state, action] = old_q + alpha * td_error

            total_reward += reward
            steps += 1

           
            # Move both state AND action forward.
            if not done: 
                state = next_state
                action = next_action

        episode_returns.append(total_reward)
        episode_steps.append(steps)

    return Q, episode_returns, episode_steps

def evaluate_policy(env, Q):
    state, info = env.reset()

    done = False
    total_reward = 0
    steps = 0
    path = [state]

    while not done:
        action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward
        steps += 1

        state = next_state
        path.append(state)

    return total_reward, steps, path

def moving_average(data, window=20):        # Add a moving average for the plot
    return np.convolve(
        data,
        np.ones(window) / window,
        mode="valid"
    )


episodes = 500      # Training Both Learning Methods
alpha = 0.5
gamma = 1.0
epsilon = 0.3
seed = 42

Q_q, q_returns, q_steps = train_q_learning(
    env,
    episodes,
    alpha,
    gamma,
    epsilon,
    seed
)

Q_sarsa, sarsa_returns, sarsa_steps = train_sarsa(
    env,
    episodes,
    alpha,
    gamma,
    epsilon,
    seed
)

q_reward, q_eval_steps, q_path = evaluate_policy(env, Q_q)      # Evaluating both methods

sarsa_reward, sarsa_eval_steps, sarsa_path = evaluate_policy(env, Q_sarsa)

print("Q-learning:")
print("Return:", q_reward)
print("Steps:", q_eval_steps)
print("Path:", q_path)

print()

print("SARSA:")
print("Return:", sarsa_reward)
print("Steps:", sarsa_eval_steps)
print("Path:", sarsa_path)

window = 20     # Graph plot

q_ma = moving_average(q_returns, window)
sarsa_ma = moving_average(sarsa_returns, window)

plt.figure(figsize=(10, 6))

plt.plot(q_ma, label="Q-learning")
plt.plot(sarsa_ma, label="SARSA")

plt.xlabel("Episode")
plt.ylabel("Moving Average Return")
plt.title("Q-learning vs SARSA on CliffWalking")
plt.legend()

plt.savefig("learning_curve_epsilon_03.png")
plt.show()
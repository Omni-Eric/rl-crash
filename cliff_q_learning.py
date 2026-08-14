import numpy as np
import gymnasium as gym

env = gym.make("CliffWalking-v0")       # Set up environment and Q table
Q = np.zeros(
    (env.observation_space.n, env.action_space.n)
)

def epsilon_greedy(Q, state, epsilon, rng):     # Policy, involves exploration
    r_number1 = rng.random()

    if r_number1 < epsilon:
        r_index = rng.integers(Q.shape[1])
        return r_index
    else:
        return np.argmax(Q[state])

def train_q_learning(       # Training Loop
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

def evaluate_q_learning(env, Q):
    state, info = env.reset()

    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward
        steps += 1
        state = next_state

    return total_reward, steps


Q, episode_returns, episode_steps = train_q_learning(       # Testing
    env=env,
    episodes=500,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
    seed=42
)

print("Training finished")
print("Q-table shape:", Q.shape)
print("Last 10 returns:", episode_returns[-10:])
print("Last 10 episode steps:", episode_steps[-10:])

reward, steps = evaluate_q_learning(env, Q)

print("Evaluation return:", reward)
print("Evaluation steps:", steps)
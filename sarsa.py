import numpy as np
import gymnasium as gym

env = gym.make("CliffWalking-v0")       # Set up environment and Q table

def epsilon_greedy(Q, state, epsilon, rng):     # Same as Q learning
    r_number = rng.random()

    if r_number < epsilon:
        return rng.integers(Q.shape[1])
    else:
        return np.argmax(Q[state])


def train_sarsa(
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


def evaluate_sarsa(env, Q):
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

Q, episode_returns, episode_steps = train_sarsa(       # Testing
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

reward, steps = evaluate_sarsa(env, Q)

print("Evaluation return:", reward)
print("Evaluation steps:", steps)
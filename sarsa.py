import numpy as np
import gymnasium as gym


def epsilon_greedy(Q, state, epsilon, rng):
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
    Q = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    rng = np.random.default_rng(seed)

    episode_returns = []
    episode_steps = []

    for i in range(episodes):
        state, info = env.reset()

        # TODO 1:
        # SARSA needs an action BEFORE entering the loop.
        action = ???

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
                # TODO 2:
                # Choose the ACTUAL next action using epsilon-greedy.
                next_action = ???

                # TODO 3:
                # SARSA uses Q[next_state, next_action]
                target = ???

            td_error = target - old_q

            Q[state, action] = old_q + alpha * td_error

            total_reward += reward
            steps += 1

            # TODO 4:
            # Move both state AND action forward.
            state = ???
            action = ???

        episode_returns.append(total_reward)
        episode_steps.append(steps)

    return Q, episode_returns, episode_steps
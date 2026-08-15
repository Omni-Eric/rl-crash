# Experiment Summary: Q-Learning vs. SARSA on CliffWalking

## Research Question

How do off-policy Q-learning and on-policy SARSA behave differently in a risky grid world when both use epsilon-greedy exploration?

## Setup

Both algorithms were implemented from scratch with NumPy and trained on Gymnasium `CliffWalking-v0`.

Main comparison settings:

- Episodes: `500`
- Learning rate: `alpha = 0.5`
- Discount factor: `gamma = 1.0`
- Random seed: `42`
- Exploration rates compared: `epsilon = 0.1` and `epsilon = 0.3`
- Learning curves: 20-episode moving-average return
- Evaluation: fully greedy with `argmax(Q[state])`

Using greedy evaluation separates the learned policy from the exploratory behavior used during training.

## Results

### Q-learning

With `epsilon = 0.3`, Q-learning learned the shortest greedy path:

- Return: `-13`
- Steps: `13`
- Path: `[36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]`

This path runs directly above the cliff.

### SARSA

With the same hyperparameters, SARSA learned a longer but safer path:

- Return: `-17`
- Steps: `17`
- Path: `[36, 24, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23, 35, 47]`

This path moves farther away from the cliff before approaching the goal.

## Effect of Exploration

Increasing epsilon from `0.1` to `0.3` made training substantially noisier for both methods, but especially for Q-learning.

Q-learning is off-policy. Its TD target uses

`reward + gamma * max(Q[next_state])`

so it learns as if the best next action will be taken, even though the behavior policy still explores randomly. This allows it to learn the shortest cliff-adjacent path, but exploratory mistakes near the cliff can produce large `-100` penalties during training.

SARSA is on-policy. Its TD target uses

`reward + gamma * Q[next_state, next_action]`

where `next_action` is actually chosen by the epsilon-greedy policy. Because exploratory actions are included in what SARSA learns, states close to the cliff become less attractive, producing a safer learned route.

## Conclusion

This experiment demonstrates a concrete difference between on-policy and off-policy temporal-difference control:

- **Q-learning** learns the shortest greedy path but can have poor training returns under continued exploration.
- **SARSA** learns a safer policy that better accounts for the risks of its exploratory behavior.

The experiment also shows that **training performance and greedy evaluation performance are different quantities**. A policy can have an excellent greedy path while still producing poor exploratory training episodes.

## Limitation and Next Step

These experiments use one random seed and 500 episodes. A stronger experimental comparison would run multiple seeds and report the mean and variability of the learning curves.

## Q-Learning Experiments

I compared three hyperparameter configurations over 500 episodes:

| Experiment | Alpha | Epsilon | Purpose |
|---|---:|---:|---|
| A | 0.1 | 0.1 | Baseline |
| B | 0.1 | 0.3 | More exploration |
| C | 0.5 | 0.1 | Higher learning rate |

![Q-Learning comparison](figures/q_learning_comparison.png)

### Results

- Experiment C learned fastest.
- Experiment B had worse training returns because it explored more frequently.
- The higher learning rate in Experiment C accelerated learning without obvious instability in this 5*5 GridWorld.
- All three experiments learned an optimal 8-step greedy policy.
- Training performance and final greedy-policy performance can differ because training uses epsilon-greedy exploration while evaluation is fully greedy.
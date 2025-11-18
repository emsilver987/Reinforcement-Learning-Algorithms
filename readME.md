# Reinforcement Learning Algorithms: L-RI vs PLA Comparison

## Overview

This project implements and compares two reinforcement learning algorithms:
- **L-RI (Linear Reward-Inaction)**: A model-free learning algorithm
- **PLA (Pursuit Learning Algorithm)**: A model-based learning algorithm

Both algorithms are evaluated in a stationary random environment with 10 actions, comparing their performance in terms of accuracy and learning speed across different learning rates.

## Problem Description

The environment simulates a learning automaton with:
- **10 actions** operating in a random but stationary environment
- **Binary feedback**: 0 (penalty) or 1 (reward)
- **Reward probabilities** for each action: `[0.19, 0.2, 0.21, 0.59, 0.6, 0.61, 0.72, 0.41, 0.39, 0.4]`
  - Action 7 (index 6) has the highest reward probability (0.72) and is the optimal action

### Learning Process

1. Action probabilities are initialized to 0.1 for all 10 actions
2. Agent selects an action using cumulative probability distribution
3. Environment generates binary feedback based on the action's reward probability
4. Agent updates action probabilities according to the chosen algorithm
5. Process continues until convergence (some probability reaches ≥ 0.9)

### Convergence Criteria

- **Convergence**: When any action probability reaches 0.9
- **Accuracy**: Convergence to the correct action (index 6 with probability 0.72)

## Algorithms

### L-RI (Linear Reward-Inaction)

- **Type**: Model-free
- **Update Rule**: Only updates on rewards, ignores penalties
  - If action `i` is chosen and rewarded:
    - `p[i] = p[i] + α(1 - p[i])`
    - `p[k] = p[k](1 - α)` for all other actions `k`
- **Characteristics**:
  - Updates do NOT accumulate
  - Reacts only to positive feedback
  - Can be slow at small learning rates
  - May become unstable at large learning rates

### PLA (Pursuit Learning Algorithm)

- **Type**: Model-based
- **Update Rule**: 
  1. Maintains Q-value estimates (running averages of reward probabilities)
  2. Identifies best action `j = argmax(Q)`
  3. Updates probabilities toward best action:
     - `p[j] = p[j] + α(1 - p[j])`
     - `p[k] = p[k](1 - α)` for all other actions `k`
- **Characteristics**:
  - Updates accumulate through Q-value estimates
  - Pursues the currently believed best action
  - Generally faster convergence than L-RI
  - More stable across learning rates

## Requirements

- Python 3.6+
- matplotlib (for plotting)
  ```bash
  pip install matplotlib
  ```

## Usage

### Running the Simulation

```bash
python RL-PLA.py
```

### What It Does

1. Runs both L-RI and PLA algorithms with 9 different learning rates: `[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]`
2. Each algorithm runs 100 times per learning rate across 5 different seeds
3. Collects metrics:
   - **Percentage Accuracy**: % of runs that converged to the correct action (index 6)
   - **Average Learning Speed**: Average number of iterations needed for convergence
4. Generates comparison graphs and saves them as `rl_pla_comparison.png`

### Output

The program will:
- Print progress for each experimental run
- Display metrics for each learning rate
- Generate and display two comparison plots:
  - **Accuracy vs Learning Rate**: Shows which algorithm achieves better accuracy
  - **Average Learning Speed vs Learning Rate**: Shows which algorithm converges faster

**Note**: The simulation may take several minutes to complete, especially at low learning rates where convergence takes many iterations.

## Project Structure

```
RL-and-PLA-Algorithms/
├── RL-PLA.py              # Main implementation file
├── readME.md              # This file
├── progass2_Fall2025.pdf  # Assignment specification
└── rl_pla_comparison.png  # Generated comparison graphs (created after running)
```

## Key Functions

- `main()`: Entry point, sets up parameters and coordinates execution
- `setUp()`: Runs experiments across all seeds and learning rates, returns averaged metrics
- `LRIUpdate()`: Implements L-RI algorithm for a single run
- `PLAUpdate()`: Implements PLA algorithm for a single run
- `check_convergence()`: Checks if any probability has reached 0.9
- `get_cumulative_arr()`: Builds cumulative probability array for action selection
- `plot_results()`: Creates comparison graphs for accuracy and learning speed

## Results & Observations

Based on the implementation and experimental results:

1. **PLA generally outperforms L-RI** in both accuracy and learning speed across most learning rates

2. **At very small learning rates** (α ≤ 0.01), both algorithms show similar performance, requiring tens of thousands of iterations to converge

3. **As learning rate increases**:
   - PLA maintains high accuracy and fast convergence
   - L-RI's accuracy begins to degrade at moderate learning rates

4. **At high learning rates** (α ≥ 0.1):
   - Both algorithms may become less stable
   - PLA's pursuit mechanism can overcommit to premature Q-estimates
   - L-RI's accuracy drops more significantly

5. **Stochastic evaluation is essential**: Running 100 repetitions per learning rate reveals true performance characteristics that single runs cannot show

## Conclusion

The project demonstrates how different reinforcement learning update rules create dramatically different learning behaviors:

- **L-RI** is simpler (reward-only updates) but slower and less stable
- **PLA** uses model-based Q-estimates to identify the best action more efficiently, resulting in faster and more reliable convergence

Both algorithms balance exploration, stability, and convergence speed differently, highlighting the trade-offs in reinforcement learning algorithm design.

## Author

This project was completed as part of Purdue's Artificial Intelligence course (CS450/CSCI 4870).

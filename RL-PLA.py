# NOTES TO TA / PROF
# The implementation takes a very long time at low learning rates but check output to see it's progress
# I have graphs which will dispaly at the end (<5 minutes) that will eval PLA vs RLI



# Need to implement L-R1 Algorithm and Pursuit Learning Algorithm (PLA)(model based)

# 10 actions should operate in random
# Environment feedback is binary (0 = penalty, 1 = reward)
# Reward probability for each ten actions
# [0.19, 0.2, 0.21, 0.59, 0.6, 0.61, 0.72, 0.41, 0.39, 0.4]

# RL Agent will operate with feedback from env given the alg
# 1. Action probabilities are initialized to 0.1
# 2. Agents calls Pseudo random Number Generator, PRN x in range 0 to 1... Formula provided. Chooses action a_i
# 3. Env generates feedback by randomly sampling reward probability d(i) for chosen action, if y < d(i) then env generates feedback 1, else 0
# 4. RL Agent updates action probabilities based on chosen RL implementation
# 5. Trials of iterations will continue until action probabilities converge, action probability reaches value of 0.9, then convergence is accurate

# L-RI and PLA RL algs run separately
# Each alg 100 times with diff random seeds
# Metrics: Percentage Accuracy(% of times RL converged to correct action), Learning speed(average # of iterations needed for convergence)

# Learning rates should be tested at different steps
# (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

# Will add plots and comments and conclusions

# L-RI 
# doesn't assume data is linearly separable, it just reacts to feedback
# updates do NOT accumulate
# uses learning rate
# IF action i is chosen and rewarded, pi := pi + (learning rate)(1-pi)
# and others pk := pk(1-(learning rate))

# PLA 
# moves decision boundary that reduces the error for a specific point, guarantees convergence if data is linearly separable
# updates accumulate
# Estimate reward probability, identify action currently believed to be best, update probabilities toward that action
# PLA update rule: pj := pj + (learning_rate)(1-pj)
# Then pk := pk - (learning rate)(pk) k cannot equal j
# This is essentially pushing our action believed "up" and all other ones down

# How action is chosen
# Probabilities given initially are part of the stationary env
# Use random number generator random.random()
# x is chosen as a random number from [0,1)
# Build a cumulative sum of the p array
# Identify what number x falls between in the cumulative sum
# This determines the action (index on stationary array) we will choose
# We will then generate another random number between [0,1]
# This will give us some random probability and if it is smaller than our simulated env index, we return 1, else 0
# y < di

# Convergence
# When any one of our probabilities (initially 0.1) reach 0.9 and IS the best action
# If the convergence is any index except 6 (which is highest at 0.72), it converged, but inaccurately

import random
import matplotlib.pyplot as plt

def main():
    reward_prob = [0.19, 0.2, 0.21, 0.59, 0.6, 0.61, 0.72, 0.41, 0.39, 0.4]
    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    seeds = [14, 12, 102, 412, 101]
    
    all_lri_metrics, all_pla_metrics = setUp(reward_prob, learning_rates, seeds)
    
    # Create plots
    plot_results(learning_rates, all_lri_metrics, all_pla_metrics)
    
def setUp(reward_prob, learning_rates, seeds):
    """
    Run experiments across all seeds and collect metrics.
    Returns averaged metrics across all seeds.
    """
    # Collect metrics from all seeds
    all_lri_metrics = []  # Will be list of [accuracy, avg_iterations] for each learning rate
    all_pla_metrics = []
    
    # Initialize accumulators for averaging across seeds
    for _ in learning_rates:
        all_lri_metrics.append([0.0, 0.0])  # [total_accuracy, total_iterations]
        all_pla_metrics.append([0.0, 0.0])
    
    # rotate through seeds - each seed represents a different experimental run
    for seed_idx in range(len(seeds)):
        print(f"\n=== Experimental Run {seed_idx + 1} (Base Seed {seeds[seed_idx]}) ===")
        metricsLRI, metricsPLA = [], []
        
        # For each learning rate, run 100 times with different seeds
        for lr_idx in range(len(learning_rates)):
            learning_rate = learning_rates[lr_idx]
            
            # Run L-RI 100 times
            lri_accurate_count = 0
            lri_total_iterations = 0
            
            for run in range(100):
                # Each run uses a different seed (100 different seeds per learning rate)
                # Use large primes to decorrelate seeds
                random.seed(seeds[seed_idx] + lr_idx * 999983 + run * 7919)
                p = [0.1] * 10
                result = LRIUpdate(reward_prob, learning_rate, p)
                if result[0]:  # converged to correct action (index 6)
                    lri_accurate_count += 1
                lri_total_iterations += result[1]
            
            lri_accuracy = lri_accurate_count / 100.0
            lri_avg_iterations = lri_total_iterations / 100.0
            metricsLRI.append([lri_accuracy, lri_avg_iterations])
            
            # Accumulate for averaging across seeds
            all_lri_metrics[lr_idx][0] += lri_accuracy
            all_lri_metrics[lr_idx][1] += lri_avg_iterations
            
            # Run PLA 100 times
            pla_accurate_count = 0
            pla_total_iterations = 0
            
            for run in range(100):
                # Each run uses a different seed (100 different seeds per learning rate)
                # Use large primes to decorrelate seeds
                random.seed(seeds[seed_idx] + lr_idx * 999983 + run * 7919)
                p = [0.1] * 10
                result = PLAUpdate(reward_prob, learning_rate, p)
                if result[0]:  # converged to correct action (index 6)
                    pla_accurate_count += 1
                pla_total_iterations += result[1]
            
            pla_accuracy = pla_accurate_count / 100.0
            pla_avg_iterations = pla_total_iterations / 100.0
            metricsPLA.append([pla_accuracy, pla_avg_iterations])
            
            # Accumulate for averaging across seeds
            all_pla_metrics[lr_idx][0] += pla_accuracy
            all_pla_metrics[lr_idx][1] += pla_avg_iterations
            
            print(f"α={learning_rate:.3f}: L-RI accuracy={lri_accuracy:.2f}, avg_iter={lri_avg_iterations:.1f} | PLA accuracy={pla_accuracy:.2f}, avg_iter={pla_avg_iterations:.1f}")
        
        print(f"\nL-RI metrics: {metricsLRI}")
        print(f"PLA metrics: {metricsPLA}")
    
    # Average across all seeds
    num_seeds = len(seeds)
    for i in range(len(learning_rates)):
        all_lri_metrics[i][0] /= num_seeds  # Average accuracy
        all_lri_metrics[i][1] /= num_seeds  # Average iterations
        all_pla_metrics[i][0] /= num_seeds
        all_pla_metrics[i][1] /= num_seeds
    
    return all_lri_metrics, all_pla_metrics

def LRIUpdate(reward_prob, learning_rate, p):
    """
    L-RI (Linear Reward-Inaction) algorithm.
    Updates only on rewards. No tracking of reward counts.
    Runs until convergence (probability >= 0.9) or max_iterations reached.
    """
    iterations = 1  # Start from 1 (first iteration/trial)
    max_iterations = 200000  # Safety upper bound, shouldn't activate in normal cases
    
    while True:
        # Choose action based on probability distribution
        rand_for_action = random.random()
        cumulative_arr = get_cumulative_arr(p)
        action_index = len(p) - 1  # default to last action
        
        for i in range(len(cumulative_arr)):
            if cumulative_arr[i] > rand_for_action:
                action_index = i
                break
        
        # Get feedback from environment
        rand_for_feedback = random.random()
        binary_reward = 1 if rand_for_feedback < reward_prob[action_index] else 0
        
        # Update probabilities ONLY on reward
        if binary_reward == 1:
            for i in range(len(p)):
                if i == action_index:
                    p[i] = p[i] + learning_rate * (1 - p[i])
                else:
                    p[i] = p[i] * (1 - learning_rate)
        
        # Normalize probabilities to prevent floating-point drift
        total = sum(p)
        if total > 0:
            for i in range(len(p)):
                p[i] = p[i] / total
        
        # Check convergence after update
        converge_index = check_convergence(p)
        if converge_index >= 0:
            # Check if converged to correct action (index 6 = action 7 with 0.72 probability)
            is_accurate = (converge_index == 6)
            return [is_accurate, iterations]
        
        iterations += 1
        
        # Safety check: prevent infinite loops
        if iterations >= max_iterations:
            return [False, iterations]

def PLAUpdate(reward_prob, learning_rate, p):
    """
    PLA (Pursuit Learning Algorithm) using running averages.
    Maintains Q estimates and updates toward best action.
    Runs until convergence (probability >= 0.9) or max_iterations reached.
    """
    # Initialize Q estimates and tracking arrays
    reward_count = [0] * len(p)
    chosen_count = [0] * len(p)
    Q = [0.0] * len(p)
    
    iterations = 1  # Start from 1 (first iteration/trial)
    max_iterations = 200000  # Safety upper bound, shouldn't activate in normal cases
    
    while True:
        # Choose action based on probability distribution
        rand_for_action = random.random()
        cumulative_arr = get_cumulative_arr(p)
        action_index = len(p) - 1  # default to last action
        
        for i in range(len(cumulative_arr)):
            if cumulative_arr[i] > rand_for_action:
                action_index = i
                break
        
        # Get feedback from environment
        rand_for_feedback = random.random()
        binary_reward = 1 if rand_for_feedback < reward_prob[action_index] else 0
        
        # Update Q estimates
        chosen_count[action_index] += 1
        if binary_reward == 1:
            reward_count[action_index] += 1
        Q[action_index] = reward_count[action_index] / chosen_count[action_index]
        
        # Find best action (argmax Q) - use max with key to avoid bias toward first equal value
        best_action = max(range(len(Q)), key=lambda i: Q[i])
        
        # Update probabilities toward best action
        for i in range(len(p)):
            if i == best_action:
                p[i] = p[i] + learning_rate * (1 - p[i])
            else:
                p[i] = p[i] * (1 - learning_rate)
        
        # Normalize probabilities to prevent floating-point drift
        total = sum(p)
        if total > 0:
            for i in range(len(p)):
                p[i] = p[i] / total
        
        # Check convergence after update
        converge_index = check_convergence(p)
        if converge_index >= 0:
            # Check if converged to correct action (index 6 = action 7 with 0.72 probability)
            is_accurate = (converge_index == 6)
            return [is_accurate, iterations]
        
        iterations += 1
        
        # Safety check: prevent infinite loops
        if iterations >= max_iterations:
            return [False, iterations]

def check_convergence(p):
    """
    Check if any probability has reached 0.9 (convergence threshold).
    Returns the index of the converged action, or -1 if no convergence.
    """
    for i in range(len(p)):
        if p[i] >= 0.9:
            return i
    return -1

def get_cumulative_arr(p):
    """
    Build cumulative probability array for action selection.
    """
    cumulative = []
    curr = 0.0
    for prob in p:
        curr += prob
        cumulative.append(curr)
    return cumulative

def plot_results(learning_rates, lri_metrics, pla_metrics):
    """
    Create two plots comparing L-RI and PLA:
    1. Accuracy vs Learning Rate
    2. Average Learning Speed (iterations) vs Learning Rate
    """
    # Extract data for plotting
    lri_accuracy = [m[0] for m in lri_metrics]
    lri_iterations = [m[1] for m in lri_metrics]
    pla_accuracy = [m[0] for m in pla_metrics]
    pla_iterations = [m[1] for m in pla_metrics]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Learning Rate
    ax1.plot(learning_rates, lri_accuracy, 'o-', label='L-RI', linewidth=2, markersize=8)
    ax1.plot(learning_rates, pla_accuracy, 's-', label='PLA', linewidth=2, markersize=8)
    ax1.set_xlabel('Learning Rate (α)', fontsize=12)
    ax1.set_ylabel('Accuracy (Percentage)', fontsize=12)
    ax1.set_title('Accuracy vs Learning Rate', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Average Learning Speed (Iterations) vs Learning Rate
    ax2.plot(learning_rates, lri_iterations, 'o-', label='L-RI', linewidth=2, markersize=8)
    ax2.plot(learning_rates, pla_iterations, 's-', label='PLA', linewidth=2, markersize=8)
    ax2.set_xlabel('Learning Rate (α)', fontsize=12)
    ax2.set_ylabel('Average Learning Speed (Iterations)', fontsize=12)
    ax2.set_title('Average Learning Speed vs Learning Rate', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('rl_pla_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphs saved as 'rl_pla_comparison.png'")
    plt.show()

if __name__ == '__main__':
    main()


# Conclusion
# Through implementing and simulating both the Linear Reward–Inaction (L-RI) algorithm and the Pursuit Learning Algorithm (PLA), I learned how dramatically different reinforcement learning behaviors emerge from the structure of the update rule itself. 
# L-RI, which only updates on rewards and ignores penalties, proved to be extremely slow at small learning rates and highly unstable at large ones. 
# It consistently required tens of thousands of iterations to converge when α was small, and its accuracy collapsed once α became too aggressive. 
# PLA, in contrast, leveraged a model-based estimate of reward probabilities, allowing it to identify the best action far more reliably and with an order-of-magnitude faster convergence. 
# However, I also observed that PLA becomes sensitive to higher learning rates: once α exceeds moderate values, its pursuit update overcommits to premature Q-estimates and accuracy drops sharply. 
# Running each algorithm 100 times per learning rate also made it clear why stochastic evaluation is essential—single runs are misleading, and only averaged accuracy and learning-speed metrics reveal the true performance characteristics. 
# Overall, this project showed me how learning automata balance exploration, stability, and convergence speed, and how subtle differences in update dynamics create very different learning profiles across the same environment.

# Graphs
# If you look atht eh grpahs, you will see that, in general PLA is far more optimal than L-RI
# for both accurarcy and speed across various learning rates
# You will notice that when the learning rate is very small for both PLA and L-RI, their perforamce is basically the same
# However, when we increase the learning rate L-RI lags behind PLA very much
# In all cases PLA has a faster average learning speed and accuracy

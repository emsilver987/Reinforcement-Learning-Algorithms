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

def main():
    reward_prob = [0.19, 0.2, 0.21, 0.59, 0.6, 0.61, 0.72, 0.41, 0.39, 0.4]
    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    seeds = [14, 12, 102, 412, 101]
    
    setUp(reward_prob, learning_rates, seeds)
    
    # graphing
    
def setUp(reward_prob, learning_rates, seeds):
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
                random.seed(seeds[seed_idx] * 10000 + lr_idx * 100 + run)
                p = [0.1] * 10
                result = LRIUpdate(reward_prob, learning_rate, p)
                if result[0]:  # converged to correct action (index 6)
                    lri_accurate_count += 1
                lri_total_iterations += result[1]
            
            lri_accuracy = lri_accurate_count / 100.0
            lri_avg_iterations = lri_total_iterations / 100.0
            metricsLRI.append([lri_accuracy, lri_avg_iterations])
            
            # Run PLA 100 times
            pla_accurate_count = 0
            pla_total_iterations = 0
            
            for run in range(100):
                # Each run uses a different seed (100 different seeds per learning rate)
                random.seed(seeds[seed_idx] * 10000 + lr_idx * 100 + run)
                p = [0.1] * 10
                result = PLAUpdate(reward_prob, learning_rate, p)
                if result[0]:  # converged to correct action (index 6)
                    pla_accurate_count += 1
                pla_total_iterations += result[1]
            
            pla_accuracy = pla_accurate_count / 100.0
            pla_avg_iterations = pla_total_iterations / 100.0
            metricsPLA.append([pla_accuracy, pla_avg_iterations])
            
            print(f"α={learning_rate:.3f}: L-RI accuracy={lri_accuracy:.2f}, avg_iter={lri_avg_iterations:.1f} | PLA accuracy={pla_accuracy:.2f}, avg_iter={pla_avg_iterations:.1f}")
        
        print(f"\nL-RI metrics: {metricsLRI}")
        print(f"PLA metrics: {metricsPLA}")

def LRIUpdate(reward_prob, learning_rate, p):
    """
    L-RI (Linear Reward-Inaction) algorithm.
    Updates only on rewards. No tracking of reward counts.
    """
    iterations = 1  # Start from 1 (first iteration/trial)
    max_iterations = 100  # Run for 100 iterations 
    
    while iterations <= max_iterations:
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
        
        # Check convergence after update
        converge_index = check_convergence(p)
        if converge_index >= 0:
            # Check if converged to correct action (index 6 = action 7 with 0.72 probability)
            is_accurate = (converge_index == 6)
            return [is_accurate, iterations]
        
        iterations += 1
    
    # Did not converge within 100 iterations
    return [False, iterations]

def PLAUpdate(reward_prob, learning_rate, p):
    """
    PLA (Pursuit Learning Algorithm) using running averages.
    Maintains Q estimates and updates toward best action.
    """
    # Initialize Q estimates and tracking arrays
    reward_count = [0] * len(p)
    chosen_count = [0] * len(p)
    Q = [0.0] * len(p)
    
    iterations = 1  # Start from 1 (first iteration/trial)
    max_iterations = 100  # Run for 100 iterations
    
    while iterations <= max_iterations:
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
        
        # Find best action (argmax Q)
        max_Q = max(Q)
        best_action = Q.index(max_Q)
        
        # Update probabilities toward best action
        for i in range(len(p)):
            if i == best_action:
                p[i] = p[i] + learning_rate * (1 - p[i])
            else:
                p[i] = p[i] * (1 - learning_rate)
        
        # Check convergence after update
        converge_index = check_convergence(p)
        if converge_index >= 0:
            # Check if converged to correct action (index 6 = action 7 with 0.72 probability)
            is_accurate = (converge_index == 6)
            return [is_accurate, iterations]
        
        iterations += 1
    
    # Did not converge within 100 iterations
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

if __name__ == '__main__':
    main()

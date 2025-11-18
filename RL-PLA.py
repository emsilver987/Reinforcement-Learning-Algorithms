# Need to implement L-R1 Algorihtm and Pursuit Learning Algorithm (PLA)(model based)

# 10 actions should operate in random
# Enviornment feedback is binary (0 = pentaly, 1 = reward)
# Reward probability for each ten actions
# [0.19, 0.2, 0.21, 0.59, 0.6, 0.61, 0.72, 0.41, 0.39, 0.4]

# RL Agent will operate with feedback from env given the alg
# 1. Action probabilites are intailized to 0.1
# 2. Agents calls Pseduo random Number Generator, PRN x in range 0 to 1... Formula provided. Chooses action a_i
# 3. Env generates feedback by randomly sampling reward probability d(i) for chosen action, if y < d(i) then env generates feddback 1, else 0
# 4. RL Agent updates action probailites based on chosen RL implementation
# 5. Trails of iterations will conitnue until action proabilites converge, action probaility reaches value of 0.9, then convergence is accurate

# L-RI and PLA RL algs run separetly
# Each alg 100 times with diff random seeds
# Metrics: Pecentage Accuracy(% of times RL converged to correct action), Learning speed(average # of interations needed for convergence)

# Learning rates should be testsed at different steps
# (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

# Will add plots and comments and conclusions

# Soultion:
# Main function will be responsible for running functions 5 times with different random seed
# Two sub-functions include RL and PRA, both running 100 times (iterating over different learning steps)
# Each of these functions will return an array of metrics [Percetange of Accuray, Learning Speed]
# These arrays will be appened to appropriate learning algoirtm forming a 2D array
# Both the RL and PRA will return these arrays up the chain, into our main function
# Now our main function will have access to each array and can build arrays for comparisons 
# We will then graph using matplotlib or some other graphing tool

# L-RI 
# doesn't assume data is linearly separable, it just reacts to feedback
# updates do NOT accumulate
# uses learning rate
# IF action i is chosen and rewarded, pi := pi + (learning rate)(1-pi)
# and others pk := pk(1-(learning rate))

# PLA 
# moves deciison boundry that reduces the error for a specific point, guarntees convergence is data is linearly separable
# updates accumulate
# Estimate reward probability, identify action currently believed to be best, update probilities toward that action
# PLA update rule: pj := pj + (learning_rate)(1-pj)
# Then pk := pk - (learning rate)(pk) k cannot equal j
# This is essentially pushing our action believed "up" and all other ones down

# How action is chosen
# Probabilites given initally are part of the stationary env
# Use random number generator random.random()
# x is chosen as a random number from [0,1)
# Build a cumulative sum of the p array
# Identify what number x falls between in the cumulative sum
# This determines the action (index on stationary array) we will choose
# We will then generate another random numebr between [0,1]
# This will give us some random probaility and if it is smaller than our simutlated env index, we return 1, else 0
# y < di

# Convergence
# When any one of our probailites (initally 0.1) reach 0.9 and IS the is the best action
# If the converegence is any index expect 7 (which is highest at 0.72), it converged, but inaccurately

import random
import re 

def main():
    reward_prob = [0.19, 0.2, 0.21, 0.59, 0.6, 0.61, 0.72, 0.41, 0.39, 0.4]
    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    seeds = [14, 12, 102, 412, 101]
    metrics = []

    setUp(reward_prob, learning_rates, seeds)

    # graphing
    

def setUp(reward_prob, learning_rates, seeds):
    # rotate through seeds
    for i in range(len(seeds)):
        # Initalization, seed
        random.seed(seeds[i])
        metricsLRI, metricsPLA = [], []

        # Iterations - reinitialize arrays for each learning rate
        for k in range(len(learning_rates)):
            LRI = [0.1] * 10
            PLA = [0.1] * 10
            metricsLRI.append(LRIUpdate(reward_prob, learning_rates[k], LRI))
            metricsPLA.append(PLAUpdate(reward_prob, learning_rates[k], PLA))

        print(metricsLRI)
        print(metricsPLA)

def LRIUpdate(reward_prob, learning, arr):
    k = 0
    while True:
        randForAction = random.random()
        cumulativeArr = getCumulativeArr(arr)
        actionIndex = 0  # default to first action
        for i in range(len(cumulativeArr)):
            if cumulativeArr[i] > randForAction:
                actionIndex = i
                break
        probOfChosenAction = random.random()
        binary = 1 if probOfChosenAction < reward_prob[actionIndex] else 0
        if binary == 1:
            for i in range(len(arr)):
                if i == actionIndex:
                    arr[i] = arr[i] + learning * (1 - arr[i])
                else: 
                    arr[i] = (1 - learning) * arr[i]
        # do nothing if binary is 0, it only learns from rewards

        convergeIndex = checkConvergence(reward_prob, arr)
        if convergeIndex > -1:
            # Accuracy: did it converge to index 6 (best action with 0.72)?
            isAccurate = (convergeIndex == 6)
            return [isAccurate, k]
        k += 1

def PLAUpdate(reward_prob, learning, arr):
    # initialize reward count, chosen count, and Q estimates
    rewardCount = [0] * len(arr)
    chosenCount = [0] * len(arr)
    Q = [0.0] * len(arr)

    k = 0
    while True:
        randForAction = random.random()
        cumulativeArr = getCumulativeArr(arr)
        actionIndex = 0  # default to first action
        for l in range(len(cumulativeArr)):
            if cumulativeArr[l] > randForAction:
                actionIndex = l
                break
        probOfChosenAction = random.random()
        binary = 1 if probOfChosenAction < reward_prob[actionIndex] else 0

        # update PLA chosen and reward and Q
        chosenCount[actionIndex] = chosenCount[actionIndex] + 1
        if binary == 1:
            rewardCount[actionIndex] = rewardCount[actionIndex] + 1
        Q[actionIndex] = rewardCount[actionIndex] / chosenCount[actionIndex]

        # update values - find best action j = argmax(Q)
        maxQ = max(Q)
        j = Q.index(maxQ)  # argmax(Q)
        for f in range(len(arr)):
            if f == j:
                arr[f] = arr[f] + (learning * (1 - arr[f]))
            else:
                arr[f] = (1 - learning) * arr[f]
        
        convergeIndex = checkConvergence(reward_prob, arr)
        if convergeIndex > -1:
            # Accuracy: did it converge to index 6 (best action with 0.72)?
            isAccurate = (convergeIndex == 6)
            return [isAccurate, k]
        k += 1


def checkConvergence(reward_prob, arr):
    for i in range(len(arr)):
        if arr[i] >= 0.9:
            return i
    return -1




def getCumulativeArr(arr):
    # funciton that provides a cumulative Arr
    curr = 0
    returnArr = []
    for i in range(len(arr)):
        curr += arr[i]
        returnArr.append(curr)
    return returnArr
    

        

if __name__ == '__main__':
    main()



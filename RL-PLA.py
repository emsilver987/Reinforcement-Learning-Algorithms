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





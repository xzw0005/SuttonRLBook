'''
Created on Sep 15, 2017

@author: wangxing
'''
import numpy as np
import matplotlib.pyplot as plt

numStates = 5                       # number of non-absorbing states (B, C, D, E, F)
states = np.arange(1, numStates+1)  # using numbers to denote non-absorbing states: B:1, C:2, D:3, E:4, F:5
startState = numStates/2 + 1        # random walk starts from middle: D (i.e., 3)
absorbingStates = [0, numStates+1]  # terminal states: A, G 

goLeft = 0
goRight = 1

trueValues = np.arange(numStates+2) / float(numStates+1) # true values for states: A:0, B:1/6, C:2/6, D:3/6, E:4/6, F:5/6, G:1
 
values = np.zeros(numStates+2)+0.5
values[0] = 0.
values[numStates+1] = 1.0

def RandomWalk():
    state = startState
    trajectory = [state]
    while state not in absorbingStates:
        action = np.random.binomial(1, 0.5)
        if action == goLeft:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
    if state == absorbingStates[1]:
        reward = 1
    else:
        reward = 0
    return trajectory, reward

def getHistory(episodes=10, runs=100):
    history = []
    for run in range(runs):
        trajectories = []
        rewards = []
        for ep in range(episodes):
            trajectory, reward = RandomWalk()
            trajectories.append(trajectory)
            rewards.append(reward)
        history.append((trajectories, rewards))
    return history

def TdLambda(lamb, alpha, episodes=10, runs=100):
    errors = []
    for run in range(runs):
        W = np.copy(values)
        trajectories = []
        rewards = []        
        for ep in range(episodes):
            eligibility = np.zeros(numStates+2)
            dw = np.zeros(numStates+2)
            trajectory, reward = RandomWalk(W, lamb, alpha)
            rewardSequence = [0] * len(trajectory)
            rewardSequence[-1] = reward
            for t in range(len(trajectory) - 1):
                s = trajectory[t]
                sp = trajectory[t+1]
                Rt = rewardSequence[t]
                eligibility *= lamb 
                eligibility[s] += 1
                dw[s] += (Rt + W[sp] - W[s])
            dw *= alpha
            dw = np.multiply(dw, eligibility)
            W += dw    
        rmse = np.sqrt(np.sum((trueValues-W)**2) / numStates)
        errors.append(rmse)
    return np.mean(errors), np.std(errors)
        
        


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

def TdLambdaOnline(lamb, alpha, episodes=10, runs=100):
    errors = []
    for run in range(runs):
        W = np.copy(values)     
        for ep in range(episodes):
            eligibility = np.zeros(numStates+2)
            dw = np.zeros(numStates+2)
            trajectory, reward = RandomWalk()
            rewardSequence = [0] * (len(trajectory))
            rewardSequence[-1] = reward
#             print trajectory
#             print rewardSequence
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
        rmse = np.sqrt(np.sum(np.power(trueValues[1:-1]-W[1:-1], 2)) / numStates)
        errors.append(rmse)
    return np.mean(errors), np.std(errors)

def TdLambdaOffline(history, lamb, alpha):
    errors = []
    for trajectories, rewards in history:   # iterates over each run
#         print '###################################'
        W = np.copy(values)
        for i in range(len(trajectories)):  # iterates over each episode
#             print '    ------------------------'
            eligibility = np.zeros(numStates+2)
            dw = np.zeros(numStates+2)
            trajectory = trajectories[i]
#             reward = rewards[i]
#             rewardSequence = [0] * (len(trajectory))
#             rewardSequence[-1] = reward
            for t in range(len(trajectory) - 1):
                s = trajectory[t]
                sp = trajectory[t+1]
                if sp == absorbingStates[1]:
                    R = 1
                else:
                    R = 0
                eligibility *= lamb 
                eligibility[s] += 1
                dw[s] += (R + W[sp] - W[s])
                dw *= alpha
                dw = np.multiply(dw, eligibility)
                W += dw
#             print W
        rmse = np.sqrt(np.sum(np.power(trueValues[1:-1]-W[1:-1], 2)) / numStates)
        errors.append(rmse)
    return np.mean(errors), np.std(errors)    

def TdLambdaP(history, lamb, alpha):
    errors = []
    for trajectories, rewards in history:           # iterates over each run
        print '###################################'
        W = np.copy(values)
        for i in range(len(trajectories)):          # iterates over each episode
            print '    ------------------------'
            dw = np.zeros(numStates + 2)
            trajectory = trajectories[i]
            reward = rewards[i]
            eligibility = np.zeros(numStates+2)
            for t in range(len(trajectory) - 1):    # iterates over time in each trajectory
                s = trajectory[t]
                x = np.zeros(numStates+2)
                x[s] = 1
                p = np.sum(np.multiply(W, x))
                sNext = trajectory[t+1]
                if sNext == absorbingStates[0]:
                    pNext = 0
                elif sNext == absorbingStates[1]:
                    pNext = 1
                else:
                    xNext = np.zeros(numStates+2)
                    xNext[sNext] = 1
                    pNext = np.sum(np.multiply(W, xNext))
                eligibility *= lamb 
                eligibility += xNext     
                dw += (pNext - p)*eligibility
            W += dw
            print W
        rmse = np.sqrt(np.sum(np.power(trueValues[1:-1]-W[1:-1], 2)) / numStates)
        errors.append(rmse)
    return np.mean(errors), np.std(errors)           

def figure4():
    history = getHistory()
    lambFig4 = [0., .3, .8, 1.]
    lambdas = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
#     lambdas = [.3]
    alphas = np.arange(0, .65, .05)
    errBestAlpha = []
#     alphas = [np.arange(0, .65, .05)] * (len(lambdas) - 1)
#     alphas.append(np.arange(0, .6, .1))
    plt.figure(4)
    for lamb in lambdas:
        errors = [] 
        for alpha in alphas:
            err, sd = TdLambdaOffline(history, lamb, alpha)
#             err, sd = TdLambdaP(history, lamb, alpha)
            errors.append(err)
        errBestAlpha.append(min(errors))
        print '$\lambda=$', lamb, ', RMSEs: ', errors
        if lamb in lambFig4:
            plt.plot(alphas, errors, '-o', label=r'$\lambda=$'+str(lamb))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('RMS error')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    plt.figure(5)
    plt.plot(lambdas, errBestAlpha, '-o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'Error Using Best $\alpha$')
    plt.show()
figure4()
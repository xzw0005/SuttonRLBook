'''
Created on Sep 15, 2017

@author: wangxing
'''
import numpy as np
import matplotlib.pylab as plt

numStates = 5   # number of non-absorbing states (B,C,D,E,F)
states = np.arange(1, numStates+1)
startState = numStates/2 + 1 # random walk starts from middle, D
absorbingStates = [0, numStates+1] # terminal states: A, G

goLeft = 0
goRight = 1

trueValues = np.zeros(numStates+2)
trueValues[1:numStates+1] = np.arange(1, numStates+1) / float(numStates+1)
trueValues[numStates+1] = 1.0

values = np.zeros(numStates+2)+.5
values[0] = 0
values[numStates+1] = 1.0

def td(values, alpha, batch=False):
    currentState = startState
    trajectory = [currentState]
    rewardSequence = [0]
    while currentState not in absorbingStates:
        action = np.random.binomial(1, 0.5)
        if action == goLeft:
            newState = currentState - 1
        else:
            newState = currentState + 1
        reward = 0
#         if newState == absorbingStates[1]:
#             reward = 1.0
#         else:
#             reward = 0.0
        if not batch:
            values[currentState] += alpha * (reward + values[newState] - values[currentState])
            # do not need to return values, since list is mutable
        currentState = newState
        trajectory.append(currentState)
        rewardSequence.append(reward)
    return trajectory, rewardSequence

def mc(values, alpha, batch=False):
    currentState = startState
    trajectory = [currentState]
    while currentState not in absorbingStates:
        action =np.random.binomial(1, 0.5)
        if action == goLeft:
            newState = currentState - 1 
        else:
            newState = currentState + 1
        if newState == absorbingStates[0]:
            returns = 0.0
        elif newState == absorbingStates[1]:
            returns = 1.0
        currentState = newState
        trajectory.append(currentState)
    if not batch:
        for state in trajectory[0:-1]:
            values[state] += alpha * (returns - values[state])
            # do not need to return values, since list is mutable
    return trajectory, [returns] * (len(trajectory) - 1)

def rmseTD(alpha, episodes, runs):
    #xAxis = np.arange(episodes)
    totalErrors = np.zeros(episodes)
    for run in range(runs):
        errors = []
        weights = np.copy(values)
        for i in range(episodes):
            sqErr = np.power(trueValues - weights, 2)
            mse = np.sum(sqErr) / numStates
            rmse = np.sqrt(mse)
            errors.append(rmse)
            td(weights, alpha)
        totalErrors += np.asarray(errors)
    return totalErrors / runs

def rmseMC(alpha, episodes, runs):
    totalErrors = np.zeros(episodes)
    for run in range(runs):
        errors = []
        weights = np.copy(values)
        for i in range(episodes):
            rmse = np.sqrt(np.sum((trueValues-weights)**2) / numStates)
            errors.append(rmse)
            mc(weights, alpha)
        totalErrors += np.asarray(errors)
    return totalErrors / runs 

def td_batch(alpha, episodes=100, runs=100):
    totalErrors = np.zeros(episodes)
    for run in range(runs):
        errors = []
        weights = np.copy(values)
        history = []
        rewards = []
        for ep in range(episodes):
#             print 'run: ', run, ' episode: ', ep
            trajectory, rewardSequence = td(weights, alpha, batch=True)
            history.append(trajectory)
            rewards.append(rewardSequence)
            converged = False
            while not converged:
                deltas = np.zeros(numStates+2)
                for trajectory, rewardSequence in zip(history, rewards):
                    for i in range(len(trajectory)-1):
                        deltas[trajectory[i]] += rewardSequence[i] + weights[trajectory[i+1]] - weights[trajectory[i]]
                deltas *= alpha
                if np.sum(np.abs(deltas)) < 1e-3:
                    converged = True
                weights += deltas
            sqErr = np.power(trueValues - weights, 2)
            mse = np.sum(sqErr) / numStates
            rmse = np.sqrt(mse)
            errors.append(rmse)
        totalErrors += np.asarray(errors)
    return totalErrors / runs

def mc_batch(alpha, episodes=100, runs=100):
    totalErrors = np.zeros(episodes)
    for run in range(runs):
        errors = []
        W = np.copy(values)
        history = []
        rewards = []
        for ep in range(episodes):
            trajectory, Gs = mc(W, alpha = 0.01, batch = True)
            history.append(trajectory)
            rewards.append(Gs)
            converged = False
            while not converged:
                dw = np.zeros(numStates + 2)
                for trajectory, Gs in zip(history, rewards):
                    for i in range(len(trajectory)-1):
                        dw[trajectory[i]] += Gs[i] - W[trajectory[i]]
                dw *= alpha
                if np.sum(np.abs(dw)) < 1e-3:
                    converged = True
                W += dw
            rmse = np.sqrt(np.sum((trueValues - W) ** 2) / numStates)
            errors.append(rmse)
        totalErrors += np.asarray(errors)
    return totalErrors / runs
            

def figure6_2a():
    episodes = [0, 1, 10, 100, 1000]
    plt.figure(1)
    states = np.arange(numStates+2)
    ws = np.copy(values)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(states, ws, '-o', label=str(i)+' episodes' )
        td(ws, .1)
    plt.plot(states, trueValues, '-o', label = 'true values')
    plt.xlabel('States')
    plt.legend()   
    plt.show() 
    
def figure6_2b():
    alphas_td = [.05, .1, .15]
    alphas_mc = [.01, .02,.03,.04]
    episodes = 101
    runs = 100
    plt.figure(2)
    xAxis = np.arange(episodes)
    for alpha in alphas_td:
        rmse = rmseTD(alpha, episodes, runs)
        plt.plot(xAxis, rmse, label=r'TD, $\alpha=$'+str(alpha))
    for alpha in alphas_mc:
        rmse = rmseMC(alpha, episodes, runs)
        plt.plot(xAxis, rmse, '--', label=r'MC, $\alpha=$'+str(alpha))
    plt.xlabel('episodes')
    plt.xlim(0, episodes-1)
    plt.legend()
    plt.show()
    
def figure6_3():
    alpha = .001
    episodes = 101
    runs = 100
    batchTD = td_batch(alpha, episodes, runs)
    batchMC = mc_batch(alpha, episodes, runs)
    xAxis = np.arange(episodes)
    plt.figure(3)
    plt.plot(xAxis, batchTD, label='TD')
    plt.plot(xAxis, batchMC, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.xlim(1, episodes)
#     plt.ylim(0, .25)
    plt.legend()
    plt.show()
    
# figure6_2a()
# figure6_2b()
figure6_3()
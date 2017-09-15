'''
Created on Sep 8, 2017

@author: wangxing
'''

import numpy as np
import matplotlib.pyplot as plt

numStates = 5   # number of non-absorbing states (B,C,D,E,F)
states = np.arange(1, numStates+1)
startState = numStates/2 + 1 # random walk starts from middle, D
absorbingStates = [0, numStates+1] # terminal states: A, G

actionLeft = 0
actionRight = 1

trueValues = np.zeros(numStates+2)
trueValues[1:numStates+1] = np.arange(1, numStates+1) / float(numStates+1)
trueValues[numStates+1] = 1
    
class TDlambda(object):
    '''
    classdocs
    '''
    def __init__(self, lamb, alpha, gamma=1.):
        '''
        Constructor
        '''
        self.lamb = lamb 
        self.alpha = alpha 
        self.gamma = gamma
        self.weights = np.zeros(numStates+2)
        self.weights[-1] = 1.0
        self.newEpisode()
        
    def newEpisode(self):
        self.eligibility = np.zeros(numStates+2)
        self.lastState = startState
        self.stateValue = 0.0
        
    def learn(self, state, reward):
#         self.eligibility *= (self.lamb * self.gamma)
        self.eligibility[self.lastState] += 1
        delta = reward + self.weights[state] * self.gamma - self.weights[self.lastState]
        delta *= self.alpha
        self.weights += delta * self.eligibility
        self.lastState = state

    
    
def randomWalk(valueFunction):
    valueFunction.newEpisode()
    currentState = startState
    trajectory = [currentState]
    rewardSequence = [0]
    while currentState not in absorbingStates:
        if np.random.binomial(1, 0.5) == actionLeft:
            newState = currentState - 1
        else:
            newState = currentState + 1
        reward = 0
#         if newState == numStates + 1:
#             reward = 1.
#         else:
#             reward = 0.
        valueFunction.learn(newState, reward)
        currentState = newState
        trajectory.append(currentState)
        rewardSequence.append(reward)
    return trajectory, rewardSequence
        
def rmsError(lambdas, alphas, episodes=10, runs=100):
    errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
    for run in range(runs):
        for lambIndex, lamb in zip(range(len(lambdas)), lambdas):
            for alphaIndex, alpha in zip(range(len(alphas[lambIndex])), alphas[lambIndex]):
                instance = TDlambda(lamb, alpha)
                for episode in range(episodes):
                    randomWalk(instance)
                    stateValues = [instance.weights[s] for s in states]
                    errors[lambIndex][alphaIndex] += np.sqrt(np.mean(np.power(stateValues - trueValues[1: -1], 2)))
    for err in errors:
        err /= episodes * runs 
#    print len(errors)
    plt.figure()
    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], '-o', label=r'$\lambda=$'+str(lambdas[i]))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('RMS error')
    plt.ylim(0, .75)
    plt.legend()
    plt.show()
        
def figure4():
    lambdas = [0, .3, .8, 1]
    alphas = [np.arange(0, .65, .05)] * (len(lambdas) - 1)
    alphas.append(np.arange(0, .6, .1))
#     alphas = [ np.arange(0, 1.1, 0.1),
#               np.arange(0, 0.99, 0.09),
#               np.arange(0, 0.55, 0.05),
#               np.arange(0, 0.33, 0.03),
#               np.arange(0, 0.22, 0.02),
#               np.arange(0, 0.11, 0.01),
#               np.arange(0, 0.044, 0.004)]
    
    rmsError(lambdas, alphas, episodes=10, runs=100)
    
figure4()
'''
Created on Sep 8, 2017

@author: wangxing
'''

import numpy as np
import matplotlib.pyplot as plt

numStates = 7
startState = numStates/2 +1
states = np.arange(1, numStates-1)
print states
absorbingState = [0, numStates-1]

actionLeft = 0
actionRight = 1

trueValues = np.zeros(numStates)
trueValues[1:numStates-1] = np.arange(1, numStates-1) / float(numStates-1)
trueValues[numStates-1] = 1
print trueValues[1:-1]
    
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
        self.values = np.zeros(numStates)
        self.newEpisode()
        
    def newEpisode(self):
        self.eligibility = np.zeros(numStates)
        self.lastState = startState
        self.stateValue = 0.0
        
    def learn(self, state, reward):
        self.eligibility *= (self.lamb * self.gamma)
        self.eligibility[self.lastState] += 1
        delta = reward + self.values[state] * self.gamma - self.values[self.lastState]
        delta *= self.alpha
        self.values += delta * self.eligibility
        self.lastState = state

    
    
def randomWalk(valueFunction):
    valueFunction.newEpisode()
    currentState = startState
    while currentState not in absorbingState:
        if np.random.binomial(1, 0.5) == actionLeft:
            newState = currentState - 1
        else:
            newState = currentState + 1
        if newState == 0:
            reward = -1
        elif newState == numStates - 1:
            reward = 1
        else:
            reward = 0
        valueFunction.learn(newState, reward)
        currentState = newState
        
def rmsError(lambdas, alphas, episodes=10, runs=100):
    errors = [np.zeros(len(lambdas))]
    for run in range(runs):
        for lambIndex, lamb in zip(range(len(lambdas)), lambdas):
            for alphaIndex, alpha in zip(range(len(alphas)), alphas):
                instance = TDlambda(lamb, alpha)
                for episode in range(episodes):
                    randomWalk(instance)
                    print instance.values
                    stateValues = [instance.values[s] for s in states]
                    errors[lambIndex] += np.sqrt(np.mean(np.power(stateValues - trueValues[1:-1], 2)))
    for err in errors:
        err /= episodes * runs 
#    print len(errors)
    plt.figure()
    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], '-o', label=r'$\lambda=$'+str(lambdas[i]))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('RMS error')
    plt.show()
        
def figure4():
    lambdas = [0, .1, .3, .5, .7, .9, 1]
    alphas = np.arange(0, .7, .1)
    rmsError(lambdas, alphas)
    
figure4()
'''
Created on Sep 8, 2017

@author: wangxing
'''

import numpy as np
import matplotlib.pyplot as plt

numStates = 7
startState = numStates/2 +1
states = np.arange(1, numStates+1)
absorbingState = [0, numStates+1]

actionLeft = 0
actionRight = 1

    
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
        self.values = np.zeros(numStates + 2)
        self.newEpisode()
        
    def newEpisode(self):
        self.eligibility = np.zeros(numStates + 2)
        self.lastState = startState
        self.stateValue = 0.0
        
    def learn(self, state, reward):
        self.eligibility *= (self.lamb * self.gamma)
        delta = reward + self.value[state] * self.gamma - self.values[self.lastState]
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
        elif newState == numStates + 1:
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
                for episode in episodes:
                    randomWalk(instance)
                    stateValues = instance.values
                    errors[lambIndex][alphaIndex] += np.sqrt(np.mean(np.power(stateValues - idealPredictions)))
        
def figure4():
    lambdas = [0, .1, .3, .5, .7, .9, 1]
    alphas = np.arange(0, .7, .1)
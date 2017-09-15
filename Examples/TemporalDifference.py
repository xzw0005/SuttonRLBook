'''
Created on Sep 14, 2017

@author: wangxing
'''
import numpy as np
import matplotlib.pylab as plt

numStates = 7
startState = numStates/2 +1
states = np.arange(1, numStates-1)
absorbingState = [0, numStates-1]

actionLeft = 0
actionRight = 1

trueValues = np.zeros(numStates)
trueValues[1:numStates-1] = np.arange(1, numStates-1) / float(numStates-1)
trueValues[numStates-1] = 1

class TD(object):
    '''
    classdocs
    '''
    def __init__(self, alpha, gamma=1.):
        '''
        Constructor
        '''
        self.alpha = alpha 
        self.gamma = gamma
        self.weights = np.zeros(numStates)
        self.newEpisode()
        
    def newEpisode(self):
        self.eligibility = np.zeros(numStates)
        self.lastState = startState
        self.trajectory = [startState]
        self.stateValue = 0.0
        
    def learn(self, state, reward):
        return
    
def randomWalk():
    currentState = startState
    trajectory=[currentState]
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
        currentState = newState
        trajectory.append(currentState)
        
def main():
    numStates = 7
    startState = numStates/2 +1
    states = np.arange(1, numStates-1)
    absorbingState = [0, numStates-1]
    
    actionLeft = 0
    actionRight = 1
    
    trueValues = np.zeros(numStates)
    trueValues[1:numStates-1] = np.arange(1, numStates-1) / float(numStates-1)
    trueValues[numStates-1] = 1
    
    lambs = [0.0, 0.1, 0.3, 0.5,  0.7,  0.9, 1.0]
    alphas = np.arange(0, .65, .05)    
    
    for alpha in alphas:
    
    
    errors = np.zeros(len(lambs))
    
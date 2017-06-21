'''
Created on Jun 21, 2017

@author: Xing Wang
'''

from __future__ import print_function
import numpy as np

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [WORLD_SIZE-1, 1]
B_POS = [0, 3]
B_PRIME_POS = [WORLD_SIZE-1, 3]
discount = 0.9

world = np.zeros((WORLD_SIZE, WORLD_SIZE))
actions = ['L', 'R', 'U', 'D']
actionProb = []
actionDict = dict({'L':0.25, 'R':0.25, 'U':0.25, 'D':0.25})
for i in range(0, WORLD_SIZE):
    actionProb.append([])
    for j in range(0, WORLD_SIZE):
        actionProb[i].append(actionDict)
        
nextState = []
actionReward = []
for i in range(WORLD_SIZE):
    nextState.append([])
    actionReward.append([])
    for j in range(WORLD_SIZE):
        next = dict()
        reward = dict()
        if i == 0:
            next['U'] = [i, j]
            reward['U'] = -1.0
        else:
            next['U'] = [i-1, j]
            reward['U'] = 0.0
        if i == WORLD_SIZE-1:
            next['D'] = [i, j]
            reward['D'] = -1.0
        else:
            next['D'] = [i+1, j]
            reward['D'] = 0.0
        if j == 0:
            next['L'] = [i, j]
            reward['L'] = -1.0
        else:
            next['L'] = [i, j-1]
            reward['L'] = 0.0  
        if j == WORLD_SIZE-1:
            next['R'] = [i, j]
            reward['R'] = -1.0
        else:
            next['R'] = [i, j+1]
            reward['R'] = 0.0   
        if [i, j] == A_POS:
            next['U'] =next['D'] =next['L'] =next['R'] = A_PRIME_POS
            reward['U'] = reward['D'] = reward['L'] = reward['R'] = 10.0
        if [i, j] == B_POS:
            next['U'] =next['D'] =next['L'] =next['R'] = B_PRIME_POS
            reward['U'] = reward['D'] = reward['L'] = reward['R'] = 5.0
        nextState[i].append(next)
        actionReward[i].append(reward)
        
print(nextState)
print(actionReward)

### For Example 3.8, Figure 3.5 ###
    # Bellman Equation 
while True:
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            for action in actions:
                newPosition = nextState[i][j][action]
                # Bellman Equation 
                newWorld[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
    if np.sum(np.abs(world-newWorld)) < 1e-5:
        print('Random Policy')
        print(newWorld)
        break
    world = newWorld
    
### For Example 3.12, Figure 3.8 ###
    # Bellman Optimality Equation
world = np.zeros((WORLD_SIZE, WORLD_SIZE))
while True:
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(0, WORLD_SIZE):
        for j in range(WORLD_SIZE):
            values = []
            for action in actions:
                newPosition = nextState[i][j][action]
                # Value Iteration 
                qval = actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]]
                values.append(qval)
            newWorld[i][j] = np.max(values)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Optimal Policy')
        print(newWorld)
        break
    world = newWorld
            
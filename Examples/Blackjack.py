'''
Created on Jul 7, 2017

@author: wangxing
'''
import numpy as np
import matplotlib.pyplot as plt

# Actions: Hits or Sticks
ACTION_HIT = 0      # request additional cards
ACTION_STICK = 1    # stops
actions = [ACTION_HIT, ACTION_STICK]

# Policy for Players
policyPlayer = np.zeros(22)
for i in range(12, 20):
    policyPlayer[i] = ACTION_HIT
policyPlayer[20] = ACTION_STICK
policyPlayer[21] = ACTION_STICK

def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# Function form of target policy of player 
def targetPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    return policyPlayer[playerSum]

# Function form of behavior policy of player 
def behaviorPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STICK
    return ACTION_HIT

# Policy for dealer 
policyDealer = np.zeros(22)
for i in range(12, 17):
    policyDealer[i] = ACTION_HIT
for i in range(17, 22):
    policyDealer[i] = ACTION_STICK
    

'''
Created on Sep 3, 2017

@author: wangxing
'''
import numpy as np

isBadSide = np.array([1, 1, 1, 0, 0, 0])
# isBadSide = np.array([1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0])
# isBadSide = np.array([1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0])
# isBadSide = np.array([0,1,0,0,0,0,0,1,0,1,1,1,1,0])
# isBadSide = np.array([0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,0,0,1,0])
# isBadSide = np.array([0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,0])
# isBadSide = np.array([0,0,1,0,0,1,1,1,1,0,1,1,0,0,1,1])
# isBadSide = np.array([0,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,0,0])
# isBadSide = np.array([0,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,0,0])
# isBadSide = np.array([0,1,1,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0])
# isBadSide = np.array([0,1,1,0,0,0,1,1])
# isBadSide = np.array([0,0,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,1])
# isBadSide = np.array([0,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,0,1,1,1,0])
# isBadSide = np.array([0,0,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,1,1,0])

# isBadSide = np.array([ ])



N = len(isBadSide)
print N
bankRoll = {0 : 1.0}
A = 2
ROLL, QUIT = range(A)

EV = 0
it = 0

while len(bankRoll) > 0:
    it += 1
    newBankRoll = {}
    for balance in bankRoll:
        balance = balance * 1.0
        
        ## if choose to roll
        valRoll = 0.
        possibleBalance = {}
        for sprime in range(N):
            if isBadSide[sprime] == 1:
                valRoll -= balance / N
            else:
                valRoll += (sprime + 1.0) / N
                newBalance = int(balance + sprime + 1.0)
                if newBalance in possibleBalance:
                    possibleBalance[newBalance] += bankRoll[balance] * 1.0 / N
                else:
                    possibleBalance[newBalance] = bankRoll[balance] * 1.0 / N
        print it, valRoll
        if valRoll >= 0:    # if choose to quit, immediate reward is 0
            action = ROLL
            for newBalance in possibleBalance:
                if newBalance in newBankRoll:
                    newBankRoll[newBalance] += possibleBalance[newBalance]
                else:
                    newBankRoll[newBalance] = possibleBalance[newBalance]      
            EV += valRoll * bankRoll[balance]              
        else:
            action = QUIT
        
    bankRoll = newBankRoll

print EV
'''
Created on Sep 8, 2017

@author: wangxing
'''

import numpy as np


probToState=0.5
valueEstimates=[0,3,8,2,1,2,0]
rewards=[0,0,0,4,1,1,1]

# Test 1
probToState=0.81
valueEstimates=[0.0,4.0,25.7,0.0,20.1,12.2,0.0]
rewards=[7.9,-5.1,2.5,-7.2,9.0,0.0,1.6]

# # # Test 2
# probToState=0.22
# valueEstimates=[0.0, -5.2, 0.0, 25.4, 10.6, 9.2, 12.3]
# rewards=[-2.4,0.8,4.0,2.5,8.6,-6.4,6.1]

# # Test 3
# probToState=0.64
# valueEstimates=[0.0,4.9,7.8,-2.3,25.5,-10.2,-6.5]
# rewards=[-2.4,9.6,-7.8,0.1,3.4,-2.1,7.9]


# # Problem 1
probToState=0.97
valueEstimates=[0.0,6.6,2.2,0.0,23.3,-1.8,19.4]
rewards=[4.6,6.2,0.0,3.2,5.4,-2.5,4.2]
# 
# # # Problem 2
probToState=0.8
valueEstimates=[0.0,-3.5,0.0,4.1,11.3,24.2,0.2]
rewards=[-0.2,6.8,0.1,-2.3,0.0,-0.5,2.4]
# 
# # # Problem 3
probToState=0.91
valueEstimates=[0.0,0.0,0.0,6.0,22.0,0.0,22.3]
rewards=[6.0,9.8,-2.4,-0.5,0.4,2.1,0.3]
# # 
# # # Problem 4
probToState=0.28
valueEstimates=[0.0,6.7,12.7,12.7,0.0,0.0,20.5]
rewards=[3.2,-4.2,3.8,4.7,-0.4,8.0,-0.7]
# # 
# # # Problem 5
probToState=0.96
valueEstimates=[0.0,5.2,0.0,2.3,20.9,22.6,-1.6]
rewards=[8.6,-3.7,1.8,8.3,5.4,-1.1,0.0]
#  
# # # Problem 6
probToState=0.62
valueEstimates=[0.0,0.0,2.6,19.5,4.8,0.6,0.0]
rewards = [-2.8,4.4,-1.1,1.1,-2.2,2.7,1.6]
# 
# # # Problem 7
probToState=0.37
valueEstimates=[0.0,18.0,-4.7,0.0,-4.0,24.9,0.0]
rewards=[6.8,6.7,3.2,-3.4,0.0,9.6,-5.0]
# 
# # # Problem 8
probToState=0.39
valueEstimates=[0.0,-2.4,-4.4,0.0,-3.0,23.2,0.0]
rewards=[1.3,5.0,-1.4,-3.0,5.1,-3.6,-1.4]
# 
# # # Problem 9
probToState=0.0
valueEstimates=[0.0,0.0,0.0,13.1,24.8,17.7,10.1]
rewards=[-1.6,6.0,-1.7,2.9,-2.9,5.6,-2.1]
# 
# # # Problem 10
probToState=0.47
valueEstimates=[0.0,0.0,0.0,10.1,22.3,-2.5,-4.1]
rewards=[-1.1,1.4,5.8,-1.0,5.0,-2.3,3.3]
#       


valueEstimates = np.array(valueEstimates, dtype=float)
rewards = np.array(rewards, dtype = float)

path1 = [0, 1, 3, 4, 5, 6]
path2 = [0, 2, 3, 4, 5, 6]
N = len(path1)

alpha = 1.0
gamma = 1.0

d1 = rewards[0] + gamma * valueEstimates[1] - valueEstimates[0]
d2 = rewards[1] + gamma * valueEstimates[2] - valueEstimates[0]
E1 = valueEstimates[0] + (probToState * d1 + (1.-probToState) * d2) * alpha
print E1 

d1 = -valueEstimates[0] + rewards[0] + gamma * rewards[2] + gamma**2 * valueEstimates[3]
d2 = -valueEstimates[0] + rewards[1] + gamma * rewards[3] + gamma**2 * valueEstimates[3]
E2 = valueEstimates[0] + (probToState * d1 + (1.-probToState) * d2) * alpha
print E2 

d1 = -valueEstimates[0] + rewards[0] + gamma * rewards[2] + gamma**2 * rewards[4] + gamma**3 * valueEstimates[4]
d2 = -valueEstimates[0] + rewards[1] + gamma * rewards[3] + gamma**2 * rewards[4] + gamma**3 * valueEstimates[4]
E3 = valueEstimates[0] + (probToState * d1 + (1.-probToState) * d2) * alpha
print E3

d1 = -valueEstimates[0] + rewards[0] + gamma * rewards[2] + gamma**2 * rewards[4] + gamma**3 * rewards[5] + gamma**4 * valueEstimates[5]
d2 = -valueEstimates[0] + rewards[1] + gamma * rewards[3] + gamma**2 * rewards[4] + gamma**3 * rewards[5] + gamma**4 * valueEstimates[5]
E4 = valueEstimates[0] + (probToState * d1 + (1.-probToState) * d2) * alpha
print E4

d1 = -valueEstimates[0] + rewards[0] + gamma * rewards[2] + gamma**2 * rewards[4] + gamma**3 * rewards[5] + gamma**4 * rewards[6] + gamma**5 * valueEstimates[6]
d2 = -valueEstimates[0] + rewards[1] + gamma * rewards[3] + gamma**2 * rewards[4] + gamma**3 * rewards[5] + gamma**4 * rewards[6] + gamma**5 * valueEstimates[6]
E5 = valueEstimates[0] + (probToState * d1 + (1.-probToState) * d2) * alpha
print E5

d1 = -valueEstimates[0] + rewards[0] + gamma * rewards[2] + gamma**2 * rewards[4] + gamma**3 * rewards[5] + gamma**4 * rewards[6] + gamma**5 * 0 + 0
d2 = -valueEstimates[0] + rewards[1] + gamma * rewards[3] + gamma**2 * rewards[4] + gamma**3 * rewards[5] + gamma**4 * rewards[6] + gamma**5 * 0 + 0
E6 = valueEstimates[0] + (probToState * d1 + (1.-probToState) * d2) * alpha
print E6

coeff = [E6-E5, E5-E4, E4-E3, E3-E2, E2-E1, E1-E6]
print np.roots(coeff)

# TD1 = E5
# from sympy import *
# x = Symbol('x')
# solve(E1 + x*E2 + x**2 *E3 + x**3 * E4 + x **4 * E5 - 1./(1-x) * E5, x)



# for step in range(1, N):
#     valPath1 = valueEstimates[0]
#     valPath2 = valueEstimates[0]
#     for i in range(step):
#         if i > 0:
#             d = gamma ** i
#             k1 = path1[i+1]
#             valPath1 +=  rewards[k1] * d * alpha
#             k2 = path2[i+1]
#             valPath2 += rewards[k2] * d * alpha
#     #print valPath1, valPath2
#     k1 = path1[step]
#     valPath1 += alpha * valueEstimates[k1]
#     k2 = path2[step]
#     valPath2 += alpha * valueEstimates[k2]
#     val = probToState * valPath1 + (1. - probToState) * valPath2 - alpha * valueEstimates[0]
#     print val
'''
Created on Apr 24, 2017

@author: xiaogou
'''
import numpy as np
n=0
v1=0
v2=0
delta = 1e3
while delta >= 0.05/(2*0.95):
    n += 1
    print '-------- n=%d ---------'%n
    v11 = 1.7 + .95*(.9 * v1 + .1* v2)
    v12 = 3.1 + .95*(.7 * v1 + .3 * v2)
    v1new = max(v11, v12)
    v21 = -.6 + .95*(.9 * v1 +.1 * v2)
    v22 = -.6 + .95*(.7 * v1 + .3 * v2)
    v2new = max(v21, v22)
    print 'v(1) = max{%.4f, %.4f}=%.4f'%(v11, v12, v1new)
    print 'v(2) = max{%.4f, %.4f}=%.4f'%(v21, v22, v2new)
    #print v21, v22, v2new
    print 'delta=sup(%.4f, %.4f)'%(v1new - v1, v2new - v2)
    delta =  max(np.abs(v1new - v1), np.abs(v2new - v2))
    v1 = v1new
    v2 = v2new

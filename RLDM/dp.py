'''
Created on Sep 29, 2017

@author: wangxing
'''
import time 

def fib(n):
    if n < 2:
        return n
    return fib(n - 2) + fib(n - 1)

t = time.time()
print fib(40)
print time.time() - t

m = {}
def fibm(n):
    if n in m:
        return m[n]
    m[n] = n if n < 2 else fibm(n-2)+fibm(n-1)
    return m[n]

t = time.time()
print fibm(40)
print time.time() - t

def fibdp(n):
    if n == 0:
        return 0
    prev, curr = (0, 1)
    for i in range(2, n+1):
        fib = prev + curr
        prev = curr 
        curr = fib 
    return curr

t = time.time()
print fibdp(40)
print time.time() - t
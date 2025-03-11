#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import levy_stable
import math
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor as PE
import pandas as pd

N = 751 # number of observations 
rng = 9 # N-1 for N days returns
assert N >= rng # check if conditions are correct

Nit = 1000 # number or MC iterations 
alpha, beta, gamma, delta = 1.7, 0.0, 1.0, 1.0 # stable distibution parameters

def MC(i): # MC function, i is iter number
    R10 = np.zeros(N-rng) 
    R = levy_stable.rvs(alpha, beta, gamma, delta, N) + 1 # generate random R1
    R10[0] = math.prod( R[0:rng] ) # calc first R10
    for k in range(1, N-rng):
            R10[k] = R10[k-1]/R[k-1]*R[k+rng] #recursively calc R10 from Rs
    R10 = R10 - 1
    return np.quantile(R10, 0.01)  # calc 0.01 quantile of R10 dist

start_time = time.time() #measure calc time
    
with PE() as executor: #init async calculations of MC fun
    Q = executor.map( MC, range(0,Nit) ) 
Q=list(Q)

end_time = time.time()
print(f"time used: {int(end_time - start_time)} seconds") #print calc time

df = pd.DataFrame(Q,columns = ['Q'])
df.to_csv( f"Q/Q_{Nit}_iters.csv", index = False ) # save data

#create hist
counts, bins, bars = plt.hist(
                              Q, 100, (-40000,0), density = True, alpha = 0.4, 
                              lw=1, color = 'blue', edgecolor = 'black' )
plt.xlabel('$Q$, arb. units')
plt.ylabel('Probability')
plt.savefig(f'MC_hist_{Nit}Nit.png', dpi = 2000)


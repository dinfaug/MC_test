#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import numpy as np
import math as m
from os.path import isfile, join
import re
from inspect import getsourcefile
from os.path import abspath
import os

path = 'Q/' # directory with the precalculated data
# change current dir to the dir with the script:
os.chdir(os.path.dirname (abspath(getsourcefile(lambda:0))))

#parse files names to obtained numbers of iterations:
file_names = [f for f in os.listdir(path) if isfile(join(path, f))]
regex = re.compile(r'\d+')
#vector with the numbers of iterations:
nit = np.sort([int(x[0]) for x in list( map( regex.findall, file_names ))])

nbns = 200 # number of the bins in histogram
rn = 20000 # x limit for the plots

#init vectors: 
median = np.zeros(len(nit))
qnt = np.zeros(len(nit))
mean = np.zeros(len(nit))
s = np.zeros(len(nit)) # square error
chitest = np.zeros(len(nit))
mantest = np.zeros(len(nit))
kstest = np.zeros(len(nit))

# calculation of the hist for max iter number
df = pd.read_csv(  f"Q/Q_{nit[-1]}_iters.csv", index_col = False )
counts, bins, bars = plt.hist( df, nbns, (-rn,0), density = True );

for (i,Nit) in enumerate(nit):
    df = pd.read_csv(f"Q/Q_{Nit}_iters.csv", index_col = False)
    Q = df.to_numpy()
    median[i] = np.median(Q)
    mean[i] = np.mean(Q)
    qnt[i] = np.quantile(Q, 0.75)
    s[i] = np.std(Q) # square error
    
    # calculation of the hist for the current iter number
    counts_cr, bins_cr, bars_cr = plt.hist(df, nbns, (-rn,0), density = True)
    
    # calculation the tests for current MC hist
    # in comparision with max iter number MC hist:
    chitest[i] = st.chisquare(counts_cr, counts).pvalue
    mantest[i] = st.mannwhitneyu(counts_cr, counts).pvalue
    kstest[i]  = st.kstest(counts_cr, counts).pvalue

# Figures:
plt.figure(1)
plt.plot(nit, median, lw = 2, marker='o', color = 'b')
plt.ylim((-12000,-11000))
plt.ylabel('median')
#ax.legend(numpoints=1)
plt.xlabel('number of iterations')
plt.savefig('Fig4.png', dpi = 2000,  bbox_inches='tight')

plt.figure(2)
plt.plot(nit, qnt, lw = 2, marker='o',  color = 'r')
plt.ylim((-6400,-5800))
plt.ylabel('0.75 quntile')
plt.xlabel('number of iterations')
plt.savefig('Fig5.png', dpi = 2000,  bbox_inches='tight')

plt.figure(3)
plt.plot(nit, mean, lw = 2, marker='o', color = 'b')
plt.ylabel('mean')
plt.xlabel('number of iterations')

plt.figure(4)
plt.plot(nit, s, lw = 2, marker='o', color = 'b')
plt.ylabel('S')
plt.xlabel('number of iterations')

plt.figure(5)
plt.plot(nit, chitest, lw = 2, marker='o', color = 'b')
plt.ylabel('chi')
plt.xlabel('number of iterations')

plt.figure(6)
plt.plot(nit,mantest, marker='o', color = 'b', label = 'MWtest')
plt.xlabel('number of iterations')
plt.plot(nit, kstest,  marker='o', color = 'r', label = 'KStest')
plt.ylabel('p-value')
plt.xlabel('number of iterations')
plt.ylim((0.8,1.1))
plt.savefig('Fig6.png', dpi = 2000,  bbox_inches='tight')
plt.legend()

# Estimation of iterations number:
E = np.mean(Q) 
S = np.std(Q) # square error
W = qnt[-1] # quantile fot the lagest iter number
B = (W - E)/(m.sqrt(2)*S) # coeff
nE = (S/E/0.1)**2 # iterations number for mean
nW = abs(S*B/W/0.01) # iterations number for quantile


#!/usr/bin/env python3

## In this example, we show Bayesian prediciotn based on the results from the Nested Sampling

# Importing computational model
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('./_model')
from model import *
import numpy as np

## Load data
f = open('data.json')
data = json.load(f)
IC = data["IC"]
thetaTrue = data["theta"]
tEval = np.array(data["t"])
observations = np.array(data["observations"])

f = open('_korali_result_nested/latest')
results = json.load(f)
samples = np.array(results["Results"]["Posterior Sample Database"])
numSamples,_ = samples.shape

tEnd = 5
NT = 200
M = 4000
y = np.zeros((M,NT))
t = np.linspace(0, tEnd, NT)

for i in range(M):
    idx = np.random.randint(numSamples)
    theta = samples[idx,:][:2]
    sig = samples[idx,:][2]
    out = solveLinearODEs(theta, IC, t)
    y[i,:] = out + np.random.normal(loc=0., scale=sig, size=NT)

median = np.quantile(y, q=0.5, axis=0)
qlow8 = np.quantile(y, q=0.1, axis=0)
qhi8 = np.quantile(y, q=0.9, axis=0)

qlow5 = np.quantile(y, q=0.25, axis=0)
qhi5 = np.quantile(y, q=0.75, axis=0)

mean = np.mean(y,axis=0)

## Plotting
fig = plt.figure(1)
#plt.plot(tEval, out)
plt.plot(tEval, observations, 'o', color='orange')
plt.plot(t, median, '-', color='red')
plt.plot(t, mean, '--', color='k')
plt.fill_between(t, qhi8, qlow8, alpha=0.2, color='r')
plt.fill_between(t, qhi5, qlow5, alpha=0.4, color='r')
plt.xlabel('Time')
plt.ylabel('Observations')
plt.show()
fig.savefig("bayesPredict.png")

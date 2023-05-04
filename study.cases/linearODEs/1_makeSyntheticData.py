#!/usr/bin/env py

# In this example, we create synthetic data from a linear system of two ODEs
# In the following scripts we try to infer the unkown parameter 'theta' 
# through an MLE approach and Bayesian UQ

import sys
sys.path.append('./_model')

import matplotlib.pyplot as plt
import json
import numpy as np
from model import solveLinearODEs

## Parse arguments
import argparse
parser = argparse.ArgumentParser(
                    prog='makeSyntheticData.py',
                    description='Make dnoisy date from ODE model and plot')
parser.add_argument('--N', help='Number of data.', default=20, type=int)
parser.add_argument('--noise', help='Standard deviation of Gaussian noise', default=2.0, type=float)

args = parser.parse_args()
N = args.N
noise = args.noise

## Define initial conditions, parameter and timestamps
IC = [10, 10]
theta = [0.2, 0.6]
tEval = np.linspace(0, 5, num=N)

## Solve model and add error terms
out = solveLinearODEs(theta, IC, tEval)
err = np.random.normal(loc=0., scale=noise, size=N)
obs = out+err

## Writing results to data.json
data = { 'IC' : IC, 'theta' : theta, 'noise' : noise, 't' : tEval.tolist(), 'observations' : obs.tolist() }
with open("data.json", "w") as outfile:
    outfile.write(json.dumps(data, indent=4))

## Plotting
fig = plt.figure(1)
plt.plot(tEval, out)
plt.plot(tEval, obs, 'o', color='orange')
plt.xlabel('Time')
plt.ylabel('Observations')
plt.show()
fig.savefig("data.png")

#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the likelihood of the data

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

## Define objective function
def objective(sample, IC, tEval, observations):
    theta = sample["Parameters"][:2]
    sigma = sample["Parameters"][-1]
    nData = len(observations)

    out = solveLinearODEs(theta, IC, tEval)
    sample["F(x)"] = -0.5*nData*np.log(2*np.pi*sigma**2)\
                        -0.5*np.sum((out-observations)**2)/sigma**2

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = lambda s: objective(s, IC, tEval, observations)

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Mu Value"] = 16
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-9
e["Solver"]["Termination Criteria"]["Max Generations"] = 1000

# Defining the problem's variables.
e["Variables"][0]["Name"] = "Theta0"
e["Variables"][0]["Initial Value"] = 0.
e["Variables"][0]["Initial Standard Deviation"] = 0.1

e["Variables"][1]["Name"] = "Theta1"
e["Variables"][1]["Initial Value"] = 0.
e["Variables"][1]["Initial Standard Deviation"] = 0.1

e["Variables"][2]["Name"] = "Sigma"
e["Variables"][2]["Initial Value"] = 1.0
e["Variables"][2]["Initial Standard Deviation"] = 1.0
e["Variables"][2]["Lower Bound"] = 0.0
e["Variables"][2]["Upper Bound"] = 10.0

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)

# Extracting results
theta = e["Results"]["Best Sample"]["Parameters"][:2]
sigma = e["Results"]["Best Sample"]["Parameters"][-1]

## Plotting
groundTruth= solveLinearODEs(thetaTrue, IC, tEval)
predict = solveLinearODEs(theta, IC, tEval)

fig = plt.figure(2)
plt.plot(tEval, groundTruth, 'b--')
plt.plot(tEval, observations, 'o', color='orange')
plt.plot(tEval, predict, 'r')
plt.fill_between(tEval, predict+sigma, predict-sigma, alpha=0.2, color='r')
plt.xlabel('Time')
plt.ylabel('Prediction')
plt.show()
fig.savefig("predict.png")

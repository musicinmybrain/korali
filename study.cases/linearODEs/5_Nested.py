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

## Define computational model
def model(sample, IC, tEval):
    theta = sample["Parameters"][:2]
    sigma = sample["Parameters"][-1]
    nData = len(tEval)

    out = solveLinearODEs(theta, IC, tEval)
    sample["Reference Evaluations"] = out.tolist()
    sample["Standard Deviation"] = nData*[sigma]

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = observations.tolist()
e["Problem"]["Computational Model"] = lambda s: model(s, IC, tEval)

# Configuring Nested Sampling parameters
e["Solver"]["Type"] = "Sampler/Nested"
e["Solver"]["Resampling Method"] = "Multi Ellipse"
e["Solver"]["Number Live Points"] = 1000

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -2.0
e["Distributions"][0]["Maximum"] = +2.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.
e["Distributions"][1]["Maximum"] = 10.0

# Defining the problem's variables.
e["Variables"][0]["Name"] = "Theta0"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "Theta1"
e["Variables"][1]["Prior Distribution"] = "Uniform 0"

e["Variables"][2]["Name"] = "Sigma"
e["Variables"][2]["Prior Distribution"] = "Uniform 1"

# Configuring results path
e["Console Output"]["Frequency"] = 500
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_nested'
e["File Output"]["Frequency"] = 500

# Running Korali
k.run(e)

"""
# Extracting results
theta = e["Results"]["Best Sample"]["Parameters"][:2]
sigma = e["Results"]["Best Sample"]["Parameters"][-1]

## Plotting
groundTruth= solveLinearODEs(thetaTrue, IC, tEval)
predict = solveLinearODEs(theta, IC, tEval)

fig = plt.figure(2)
plt.plot(tEval, groundTruth, 'b--')
plt.plot(tEval, observations, 'o')
plt.plot(tEval, predict, 'r')
plt.fill_between(tEval, predict+sigma, predict-sigma, alpha=0.2, color='r')
plt.xlabel('Time')
plt.ylabel('Prediction')
plt.show()
fig.savefig("predict.png")
"""

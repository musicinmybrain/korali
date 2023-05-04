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
def model(sample, IC, tEval, observations):
    theta = sample["Parameters"][:2]
    sigma = sample["Parameters"][-1]
    nData = len(observations)

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
e["Problem"]["Computational Model"] = lambda s: model(s, IC, tEval, observations)

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/MCMC"
e["Solver"]["Burn In"] = 1000
e["Solver"]["Use Adaptive Sampling"] = True
e["Solver"]["Termination Criteria"]["Max Samples"] = 10000

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -5.0
e["Distributions"][0]["Maximum"] = +5.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.
e["Distributions"][1]["Maximum"] = 10.0

# Defining the problem's variables.
e["Variables"][0]["Name"] = "Theta0"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 0.
e["Variables"][0]["Initial Standard Deviation"] = 0.25

e["Variables"][1]["Name"] = "Theta1"
e["Variables"][1]["Prior Distribution"] = "Uniform 0"
e["Variables"][1]["Initial Mean"] = 0.
e["Variables"][1]["Initial Standard Deviation"] = 0.25

e["Variables"][2]["Name"] = "Sigma"
e["Variables"][2]["Prior Distribution"] = "Uniform 1"
e["Variables"][2]["Initial Mean"] = 5.
e["Variables"][2]["Initial Standard Deviation"] = 1.0

# Configuring results path
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_mcmc'
e["File Output"]["Frequency"] = 100

# Running Korali
k.run(e)

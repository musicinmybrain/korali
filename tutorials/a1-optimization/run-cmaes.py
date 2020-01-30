#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
sys.path.append('./model')
from model import *

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Evaluation/Direct/Basic"
e["Problem"]["Objective"] = "Maximize"

# Defining the problem's variables.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
e["Solver"]["Termination Criteria"]["Max Generations"] = 10

e["Random Seed"] = 0xC0FFEE

# Loading previous results, if they exist.
found = False
#found = e.loadState()

# If not found, we run first 5 generations.
if (found == False):
 print('------------------------------------------------------')
 print('Running first 5 generations anew...')
 print('------------------------------------------------------')

# If found, we continue with the next 5 generations.
if (found == True):
 print('------------------------------------------------------')
 print('Running 5 more generations from previous run...')
 print('------------------------------------------------------')
 e["Solver"]["Termination Criteria"]["Max Generations"] = e["Solver"]["Termination Criteria"]["Max Generations"] + 5

# Running 10 generations
e["Problem"]["Objective Function"] = model
k.run(e)


#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--run', help='Run Number', required=False, type=int, default = 0)
parser.add_argument('--optimizer', help='Optimizer', required=False, type=str, default = "Adam")

parser.add_argument('--nPolicies', help='Number of Policies in Ensemble.', required=False, type=int, default = 5)
parser.add_argument('--bGaussian', help='Boolean to decide whether we use Gaussian approximation.', required=False, type=bool, default=True)
parser.add_argument('--bBayesian', help='Boolean to decide whether we use Bayesian Learning.', required=False, type=bool, default=False)
parser.add_argument('--nSamples', help='Number of Samples from Posterior that are used to approximate posterior predicitve distribution.', required=False, type=int, default = 1)
parser.add_argument('--nHyperparameters', help='Number of Hyperparameters that are stored.', required=False, type=int, default = 1)
parser.add_argument('--bSWAG', help='Boolean to decide whether we use SWAG.', required=False, type=bool, default=False)
parser.add_argument('--langevin', help='Use Langevin Dynamics.', required=False, type=bool, default=False)
parser.add_argument('--dropout', help='Dropout probability.', required=False, type=float, default=0.0)
parser.add_argument('--hmc', help='HMC Number of Steps.', required=False, type=int, default=0)

args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = "run{:02d}".format(args.run)
found = e.loadState(resultFolder + '/latest');
if found:
 print("Loading data from previous run..")
# for testing ensemble
# allHyperparameters = []
# for i in range(args.nPolicies):
# 	eOld = korali.Experiment()
# 	found = eOld.loadState("baseline/run{:02d}/latest".format(i))
# 	hyperparameters = eOld["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"][0]
# 	allHyperparameters.append(hyperparameters)
# e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"] = allHyperparameters

### Set random seed

e["Random Seed"] = args.run + 1
# for testing
# e["Random Seed"] = e["Current Generation"]
# for testing ensemble
# e["Random Seed"] = 500

## Defining probelem configurations
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : agent(x, env)
e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
e["Problem"]["Custom Settings"]["Rendering"] = "Disabled"
e["Problem"]["Testing Frequency"] = 20
e["Problem"]["Policy Testing Episodes"] = 10

### Initializing openAI Gym environment and setting environment function

initEnvironment(e, args.env, e["Random Seed"], e["Problem"]["Custom Settings"]["Rendering"] == "Enabled")
# for testing
# initEnvironment(e, args.env, 42)

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
# e["Solver"]["Mode"] = "Testing"
# e["Solver"]['Testing']['Sample Ids'] = [ 42 ]
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Settings to enable Bayesian Reinforcement Learning

e["Solver"]["Use Gaussian Approximation"] = args.bGaussian
e["Solver"]["Burn In"] = 0

# Ensemble Learning
e["Problem"]["Policies Per Environment"] = args.nPolicies
e["Problem"]["Ensemble Learning"] = args.nPolicies > 1

# Switch to enable / disable Bayesian learning (needed for all of the following options)
e["Solver"]["Bayesian Learning"] = args.bBayesian

# Set the number of samples that are used
e["Solver"]["Number Of Samples"] = args.nSamples

# Enable Dropout (https://proceedings.mlr.press/v48/gal16.html)
e["Solver"]["Dropout Probability"] = args.dropout

# Set the number of samples that are used
e["Solver"]["Number Of Stored Hyperparameters"] = args.nHyperparameters

# Enable SWAG (https://arxiv.org/pdf/1902.02476.pdf)
e["Solver"]["swag"] = args.bSWAG

# Enable Langevin Dynamics (https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
e["Solver"]["Langevin Dynamics"] = args.langevin

# Enable Hamiltonian Monte Carlo (https://proceedings.mlr.press/v48/gal16.html)
e["Solver"]["hmc"]["Mass"] = 1.0
e["Solver"]["hmc"]["Number Of Steps"] = args.hmc
e["Solver"]["hmc"]["Step Size"] = 1e-4
e["Solver"]["hmc"]["Enabled"] = args.hmc > 0

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = args.optimizer
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = 5e6
e["Solver"]["Termination Criteria"]["Max Generations"] = 500
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)

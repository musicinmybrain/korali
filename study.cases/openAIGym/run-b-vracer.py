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

parser.add_argument('--nPolicies', help='Number of Policies in Ensemble.', required=False, type=int, default = 1)

parser.add_argument('--bBayesian', help='Boolean to decide whether we use Bayesian Learning.', required=False, type=bool, default = False)
parser.add_argument('--nSGD', help='Number of Samples from Posterior that are stored.', required=False, type=int, default = 1)
parser.add_argument('--bSWAG', help='Boolean to decide whether we use SWAG.', required=False, type=int, default = 0)
parser.add_argument('--langevin', help='Weighting of gradient noise for Langevin Dynamics.', required=False, type=float, default=0.0)
parser.add_argument('--dropout', help='Dropout probability.', required=False, type=float, default=0.0)
parser.add_argument('--hmc', help='HMC Mass.', required=False, type=float, default=0.0)

args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = 'run' + str(args.run) +'/'
e.loadState(resultFolder + '/latest');

### Initializing openAI Gym environment

initEnvironment(e, args.env)

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Settings to enable Bayesian Reinforcement Learning

# Ensemble Learning
e["Problem"]["Policies Per Environment"] = args.nPolicies
e["Problem"]["Ensemble Learning"] = args.nPolicies > 1

# Posterior Sampling
e["Solver"]["Bayesian Learning"] = args.nSGD > 1
e["Solver"]["Number Of Samples"] = args.nSGD

# Enable SWAG (https://arxiv.org/pdf/1902.02476.pdf)
e["Solver"]["swag"] = args.bSWAG

# Enable Langevin Dynamics (https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
e["Solver"]["Langevin Dynamics Noise Level"] = args.langevin

# Enable Dropout (https://proceedings.mlr.press/v48/gal16.html)
e["Solver"]["Dropout Probability"] = args.dropout

# Enable Hamiltonian Monte Carlo (https://proceedings.mlr.press/v48/gal16.html)
e["Solver"]["hmc"]["Mass"] = args.hmc
e["Solver"]["hmc"]["Number Of Steps"] = 5
e["Solver"]["hmc"]["Step Size"] = 1e-4
e["Solver"]["hmc"]["Enabled"] = args.hmc > 0.0

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
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
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

e["Solver"]["Termination Criteria"]["Max Experiences"] = 10e6
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1000
# e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)

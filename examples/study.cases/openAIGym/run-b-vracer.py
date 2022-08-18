#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--dis', help='Sampling Distribution.', required=False, type=str, default = "Clipped Normal")
parser.add_argument('--l2', help='L2 Regularization.', required=False, type=float, default = 0.)
parser.add_argument('--opt', help='Off Policy Target.', required=False, type=float, default = 0.1)
parser.add_argument('--lr', help='Learning Rate.', required=False, type=float, default = 0.0001)
parser.add_argument('--nPolicies', help='Number of Policies in Ensemble.', required=False, type=int, default = 1)
parser.add_argument('--nSWAG', help='Number of Hyperparameters for SWAG.', required=False, type=int, default = 10)
parser.add_argument('--K', help='Rank of Covariance Matrix for SWAG.', required=False, type=int, default = 5)
parser.add_argument('--nSamples', help='Number of Samples from Posterior distribution.', required=False, type=int, default = 30)
parser.add_argument('--startSampling', help='Number of Epsisodes before Sampling starts.', required=False, type=int, default = 100)
args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

dis_dir = args.dis.replace(" ","_")
resultFolder = '_result_b_vracer_' + args.env + '_' + dis_dir + '_' + str(args.lr) + '_' + str(args.opt) + '_' + str(args.l2) + '/'

### Initializing openAI Gym environment

initEnvironment(e, args.env)

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = args.lr
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Settings to enable Bayesian Reinforcement Learning

e["Problem"]["Policies Per Environment"] = args.nPolicies
e["Problem"]["Ensemble Learning"] = args.nPolicies > 1
e["Solver"]["Stochastic Weight Averaging Horizon"] = args.nSWAG
e["Solver"]["Rank Of Covariance"] = args.K
e["Solver"]["Number Of Samples"] = args.nSamples
e["Solver"]["Start Sampling Generation"] = args.startSampling

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = args.opt

e["Solver"]["Policy"]["Distribution"] = args.dis
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = args.l2 > 0.
e["Solver"]["L2 Regularization"]["Importance"] = args.l2

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = 10e6
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 10
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)

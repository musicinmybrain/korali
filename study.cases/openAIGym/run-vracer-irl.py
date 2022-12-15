#!/usr/bin/env python3
import sys
sys.path.append('./_model')

import json
import argparse

import numpy as np
from agent import *

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--rnn', help='Reward Neural Net size.', required=False, default=8, type=int)
parser.add_argument('--ebru', help='Experiences between reward update.', required=False, default=500, type=int)
parser.add_argument('--dbs', help='Demonstration Batch Size.', required=False, default=5, type=int)
parser.add_argument('--bbs', help='Background Batch Size.', required=False, default=50, type=int)
parser.add_argument('--bss', help='Background Sample Size.', required=False, default=100, type=int)
parser.add_argument('--pol', help='Demonstration Policy (Constant, Linear or Quadratic).', required=False, default="Linear", type=str)
parser.add_argument('--exp', help='Number of expriences.', required=False, default=1000000, type=int)
parser.add_argument('--run', help='Run number, used for output.', type=int, required=False, default=0)

args = parser.parse_args()
print(args)

####### Load observations
excludePositions = False
if (args.env == 'Reacher-v4'):
  excludePositions = True
elif (args.env == 'Ant-v4'):
  excludePositions = True
 
obsfile = f"observations_{args.env}.json" if excludePositions else f"observations_position_{args.env}.json"
rawstates = []
rawactions = []
with open(obsfile, 'r') as infile:
    obsjson = json.load(infile)
    rawstates = obsjson["States"]
    rawactions = obsjson["Actions"]

print(f"Number of raw states {len(rawstates)}")
print(f"Number of raw actions {len(rawactions)}")

### Compute Feauters from states
obsstates = []
obsfeatures = []
obsactions = []
for tstates, tactions in zip(rawstates, rawactions):
    states = []
    features = []
    actions = []
    for idx in range(len(tstates)):
        states.append([list(tstates[idx])]) if excludePositions else states.append([list(tstates[idx][1:])])
        features.append([list(tstates[idx])])
        actions.append([list(tactions[idx])])

    obsstates.append(states)
    obsfeatures.append(features)
    obsactions.append(actions)

print("Total observed trajectories: {}/{}".format(len(obsstates), len(obsactions)))

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = f'_result_irl_{args.env}_{args.run}'
e.loadState(resultFolder + '/latest');

### Initializing openAI Gym environment

initEnvironment(e, args.env, excludePositions)

### IRL variables

e["Problem"]["Observations"]["States"] = obsstates
e["Problem"]["Observations"]["Actions"] = obsactions
e["Problem"]["Observations"]["Features"] = obsfeatures
e["Problem"]["Custom Settings"]["Print Step Information"] = "Enabled"

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 1e-9
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Feature Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False

### IRL related configuration

e["Solver"]["Demonstration Policy"] = args.pol
e["Solver"]["Optimize Max Entropy Objective"] = True
e["Solver"]["Experiences Between Reward Updates"] = args.ebru
e["Solver"]["Demonstration Batch Size"] = args.dbs
e["Solver"]["Background Batch Size"] = args.bbs
e["Solver"]["Background Sample Size"] = args.bss
e["Solver"]["Use Fusion Distribution"] = True
e["Solver"]["Experiences Between Partition Function Statistics"] = 1e5

## Reward Function Specification

e["Solver"]["Reward Function"]["Batch Size"] = 256
e["Solver"]["Reward Function"]["Learning Rate"] = 1e-4

e["Solver"]["Reward Function"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["Reward Function"]["L2 Regularization"]["Importance"] = 0.

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.rnn

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.rnn

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"


### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.exp

### Setting file output configuration

e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 200
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)
print(args)

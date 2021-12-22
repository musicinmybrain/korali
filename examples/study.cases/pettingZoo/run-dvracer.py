#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *
import pdb

####### Parsing arguments

os.environ["SDL_VIDEODRIVER"] = "dummy"
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--l2', help='L2 Regularization.', required=False, type=float, default = 0.)
parser.add_argument('--opt', help='Off Policy Target.', required=False, type=float, default = 0.1)
parser.add_argument('--lr', help='Learning Rate.', required=False, type=float, default = 0.0001)
parser.add_argument('--nn', help='Neural net width of two hidden layers.', required=False, type=int, default = 128)
parser.add_argument('--run', help='Run Number', required=False, type=int, default = 0)
parser.add_argument('--multpolicies', help='If set to 1, train with N policies', required=False, type=int, default = 0)
parser.add_argument('--model', help='Model Number', required=False, type=str, default = '')
parser.add_argument('--exp', help='Max experiences', required=False, type=int, default = 10000000)
#model '0' or '' weakly Dependent Individualist 
#model '1' strongly Dependent Individualist
#model '2' weakly Dependent Collectivist  
#model '3' strongly Dependent Collectivist 
#model '4' Baseline (Individual)

args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

# resultFolder = 'results/_result_dvracer_' + args.env + '_' + str(args.nn)+ '_'+ str(args.run) +'/'
resultFolder = 'run' + str(args.run) +'/'
e.loadState(resultFolder + '/latest');

### Initializing openAI Gym environment

initEnvironment(e, args.env, args.multpolicies)

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Discrete / dVRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = args.lr
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256
e["Solver"]["Multi Agent Relationship"] = 'Individual'
e["Solver"]["Multi Agent Correlation"] = False

if(args.model == '1'):
	e["Solver"]["Multi Agent Correlation"] = True

elif(args.model == '2'):
	e["Solver"]["Multi Agent Relationship"] = 'Cooperation'

elif(args.model == '3'):
	e["Solver"]["Multi Agent Relationship"] = 'Cooperation'
	e["Solver"]["Multi Agent Correlation"] = True

elif(args.model == '4'):
	e["Solver"]["Multi Agent Sampling"] = "Baseline"

### Setting Experience Replay and REFER settings

if (args.env == 'Pursuit'):
	numAg = 8
elif (args.env == 'Gather'):
	numAg = 495
else:
	print("Environment '{}' not recognized! Exit..".format(args.env))
	sys.exit()

# e["Solver"]["Experience Replay"]["Start Size"] = 256
e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = args.opt


e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = args.l2 > 0.
e["Solver"]["L2 Regularization"]["Importance"] = args.l2

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.nn

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.nn

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.exp
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 5
#e["File Output"]["Frequency"] = 2
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)

file = open(resultFolder + 'args.txt',"w")
file.write(str(args))
file.close()

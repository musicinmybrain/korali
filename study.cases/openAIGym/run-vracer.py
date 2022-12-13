import sys
import argparse
sys.path.append('_model')
from environment import *

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--dis', help='Sampling Distribution.', required=False, type=str, default="Clipped Normal")
parser.add_argument('--l2', help='L2 Regularization.', required=False, type=float, default = 0.)
parser.add_argument('--opt', help='Off Policy Target.', required=False, type=float, default = 0.1)
parser.add_argument('--lr', help='Learning Rate.', required=False, type=float, default = 0.0001)
parser.add_argument('--exp', help='Number of experiences to run.', required=False, type=int, default = 1000000)
parser.add_argument('--run', help='Run tag.', required=False, type=int, default = 0)
parser.add_argument('--test', help='Run policy evaluation.', required=False, action='store_true')
parser.add_argument('--n', help='Number of trajectories.', required=False, type=int, default=100)
args = parser.parse_args()
print(args)

excludePos = False

####### Defining Korali Problem

import korali

k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

dis_dir = args.dis.replace(" ","_")
resultFolder = f'_result_vracer_{args.env}_{args.run}/'
e.loadState(resultFolder + '/latest')

### Set random seed

e["Random Seed"] = 0xC0FEE

### Initializing openAI Gym environment

initEnvironment(e, args.env, excludePos)

e["Problem"]["Testing Frequency"] = 10
e["Problem"]["Policy Testing Episodes"] = 50

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if args.test else "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = args.lr
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = args.opt

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = args.l2 > 0.
e["Solver"]["L2 Regularization"]["Importance"] = args.l2

e["Solver"]["Policy"]["Distribution"] = args.dis
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.exp
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 200
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Path"] = resultFolder

### Running Experiment
if args.test:
    e["Solver"]["Testing"]["Sample Ids"] = list(range(args.n))
    e["Problem"]["Custom Settings"]["Save State"] = "True"
    e["Problem"]["Custom Settings"]["File Name"] = f"observations_{args.env}.json" if excludePositions else f"observations_position_{args.env}.json"
 
k.run(e)

#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from double_env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Pendulum problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 750
e["Problem"]["Policy Testing Episodes"] = 10
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][2]["Name"] = "Angle 1"
e["Variables"][3]["Name"] = "Angular Velocity 1"
e["Variables"][4]["Name"] = "Angle 2"
e["Variables"][5]["Name"] = "Angular Velocity 2"
e["Variables"][6]["Name"] = "Height Proxy"

e["Variables"][7]["Name"] = "Force"
e["Variables"][7]["Type"] = "Action"
e["Variables"][7]["Lower Bound"] = -20.0
e["Variables"][7]["Upper Bound"] = +20.0
e["Variables"][7]["Initial Exploration Noise"] = 2.00

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / DDPG"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 10
e["Solver"]["Learning Rate"] = 0.001
e["Solver"]["Policy"]["Learning Rate Scale"] = 0.1
e["Solver"]["Policy"]["Adoption Rate"] = 0.1

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 65536

### Configuring the Remember-and-Forget Experience Replay algorithm

e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5e-7
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3

### Configuring Mini Batch

e["Solver"]["Mini Batch Size"] = 128
e["Solver"]["Mini Batch Strategy"] = "Uniform"

### Configuring the neural network and its hidden layers

e["Solver"]["Time Sequence Length"] = 1

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 900

### Setting file output configuration

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
 
### Running Training Experiment

k.run(e)

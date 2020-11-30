#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Angular Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Force"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -10.0
e["Variables"][4]["Upper Bound"] = +10.0
e["Variables"][4]["Exploration Sigma"] = 0.35

### Configuring ACER hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / CACER"
e["Solver"]["Importance Weight Truncation"] = 2.0
e["Solver"]["Optimization Steps Per Update"] = 1
e["Solver"]["Experiences Between Agent Trainings"] = 1

e["Solver"]["Random Action Probability"]["Initial Value"] = 0.5
e["Solver"]["Random Action Probability"]["Target Value"] = 0.01
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.03

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

## Defining Q-Critic and Action-selection (policy) optimizers

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 0.001
e["Solver"]["Critic"]["Mini Batch Size"] = 32
e["Solver"]["Critic"]["Retrace"]["Enabled"] = True
e["Solver"]["Critic"]["Retrace"]["Cache Persistence"] = 5

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"

## Defining Policy Configuration

e["Solver"]["Policy"]["Learning Rate"] = 0.001
e["Solver"]["Policy"]["Mini Batch Size"] = 32

e["Solver"]["Policy"]["Trust Region"]["Enabled"] = True
e["Solver"]["Policy"]["Trust Region"]["Divergence Constraint"] = 0.5
e["Solver"]["Policy"]["Trust Region"]["Adoption Rate"] = 0.9

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Output"]["Scale"] = [ 10.0 ]

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 400
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 450

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

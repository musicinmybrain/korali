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
e["Problem"]["Actions Between Policy Updates"] = 1
e["Problem"]["Training Reward Threshold"] = 3.0
e["Problem"]["Policy Testing Episodes"] = 10

### Defining State variables

e["Variables"][0]["Name"] = "Position X"
e["Variables"][1]["Name"] = "Position Y"

### Defining Action variables 

e["Variables"][2]["Name"] = "Force X"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Lower Bound"] = -0.1
e["Variables"][2]["Upper Bound"] = +0.1
e["Variables"][2]["Initial Exploration Noise"] = 0.08

e["Variables"][3]["Name"] = "Force Y"
e["Variables"][3]["Type"] = "Action"
e["Variables"][3]["Lower Bound"] = +0.0
e["Variables"][3]["Upper Bound"] = +1.0
e["Variables"][3]["Initial Exploration Noise"] = 0.24

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Mini Batch"]["Size"] = 256

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Time Sequence Length"] = 4

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16 

#e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
#e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

#e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
#e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

#e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
#e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

#e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
#e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 2.0

### Setting file output configuration

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
 
### Running Training Experiment

k.run(e)

#lander = Lander()
#lander.reset()
#done = False
#while not done:
 #state = lander.getState()
 #print(state)
 #done = lander.advance([0.0, 0.0])

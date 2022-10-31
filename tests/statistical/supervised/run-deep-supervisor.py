#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import korali
k = korali.Engine()

trainingBatchSize = 20
testBatchSize = 10
testMSEThreshold = 0.05
scaling = 5.0
np.random.seed(0xC0FFEE)

# The input set has scaling and a linear element to break symmetry
trainingInputSet = np.random.uniform(0, 2 * np.pi, trainingBatchSize)
trainingSolutionSet = np.tanh(np.exp(np.sin(trainingInputSet))) * scaling 

trainingInputSet = [ [ [ i ] ] for i in trainingInputSet.tolist() ]
trainingSolutionSet = [ [ i ] for i in trainingSolutionSet.tolist() ]

### Defining a learning problem to infer values of sin(x)

e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Testing Batch Size"] = testBatchSize

e["Problem"]["Input"]["Data"] = trainingInputSet
e["Problem"]["Input"]["Size"] = 1
e["Problem"]["Solution"]["Data"] = trainingSolutionSet
e["Problem"]["Solution"]["Size"] = 1

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "DeepSupervisor"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Learning Rate"] = 0.005
e["Solver"]["L2 Regularization"]["Enabled"] = True
e["Solver"]["L2 Regularization"]["Importance"] = 0.05
e["Solver"]["Termination Criteria"]["Target Loss"] = testMSEThreshold * 0.2

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "Korali"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 8

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 8

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Output Activation"] = "Elementwise/Tanh"
e["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 5.0
e["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0
e["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity"
    
### Configuring output

e["Console Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = 1000
k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = np.random.uniform(0, 2 * np.pi, testBatchSize)
testInputSet = [ [ [ i ] ] for i in testInputSet.tolist() ]
testOutputSet = [ x[0][0] for x in np.tanh(np.exp(np.sin(testInputSet))) * scaling ]

e["Solver"]["Mode"] = "Testing"
e["Problem"]["Input"]["Data"] = testInputSet

### Running Testing and getting results
k.run(e)
testInferredSet = [ x[0] for x in e["Solver"]["Evaluation"] ]

    
### Calc MSE on test set
mse = np.mean((np.array(testInferredSet) - np.array(testOutputSet))**2)
print("MSE on test set: {}".format(mse))

if (mse > 0.1):
 print("Fail: MSE does not satisfy threshold: " + str(0.1))
 exit(-1)

 ### Plotting Results

 #if (args.plot):
 # plt.plot(testInputSet, testOutputSet, "o")
 # plt.plot(testInputSet, testInferredSet, "x")
 # plt.show()

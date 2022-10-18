#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
from mnist import MNIST

k = korali.Engine()
e = korali.Experiment()

### Hyperparameters

### Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]
mndata = MNIST('./_data')
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()
img_width = 28
img_height = 28
### Interpolation size
img_height_red = img_height/2
img_width_red = img_width/2
example_image_idx = 0
### Confirm correct dimensions img_width*img_height == flattend_img_length
assert img_width*img_height == len(trainingImages[example_image_idx])
### Converting images to Korali form (requires a time dimension)
trainingImageVector = [ [trainingImages[example_image_idx]] ]
input_size = output_size = img_width*img_height
### Configuring general problem settings
k["Conduit"]["Type"] = "Sequential"
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = 1
e["Problem"]["Input"]["Size"] = input_size
### TODO make unsupervisded learning possible without solution and mode
e["Problem"]["Solution"]["Size"] = output_size
e["Solver"]["Mode"] = "Training"
### Using a neural network solver (deep learning) for inference
e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# ===================== Input Layer
e["Problem"]["Input"]["Size"] = input_size
# ===================== Down Sampling
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Resampling Type"] = "Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Resampling"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"] = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"] = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Width"] = img_width_red
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Height"] = img_height_red
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = img_height_red*img_width_red
# ===================== Store Output Layer

# ===================== Up Sampling
e["Solver"]["Neural Network"]["Output Layer"]["Resampling Type"] = "Linear"
e["Solver"]["Neural Network"]["Output Layer"]["Type"] = "Layer/Resampling"
e["Solver"]["Neural Network"]["Output Layer"]["Image Width"] = img_width_red
e["Solver"]["Neural Network"]["Output Layer"]["Image Height"] = img_height_red
e["Solver"]["Neural Network"]["Output Layer"]["Output Width"] = img_width
e["Solver"]["Neural Network"]["Output Layer"]["Output Height"] = img_height
e["Solver"]["Neural Network"]["Output Layer"]["Output Channels"] = img_width*img_height
### Printing Configuration

print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Epochs: " + str(epochs))
print("[Korali] Initial Learning Rate: " + str(learningRate))
print("[Korali] Decay: " + str(decay))

### Running the resampling
e["Solver"]["Mode"] = "Training"
e["Problem"]["Input"]["Data"] = trainingImageVector
e["Problem"]["Solution"]["Data"] = [ x[0] for x in trainingImageVector]
k.run(e)

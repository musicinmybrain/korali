#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
sys.path.append(os.path.abspath('./_models'))
from mnist import MNIST

# utilities

from autoencoder import configure_autencoder
add_time_dimension = lambda l : [ [y] for y in l]

# Setup Korali

k = korali.Engine()
e = korali.Experiment()

# Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]

mndata = MNIST("./_data")
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _  = mndata.load_testing()
img_size = len(trainingImages[0])

# Converting images to Korali form (requires a time dimension)

trainingTargets = trainingImages
trainingImages  = add_time_dimension(trainingImages)
testingImages   = add_time_dimension(testingImages)

img_width      = 28
img_height     = 28
input_channels = 1
latent_dims    = 10
batch_size     = 128

stepsPerEpoch = int(len(trainingImages) / batch_size)
testingBatchSize = len(testingImages)

# Configure Conduit

k["Conduit"]["Type"] = "Sequential"

# Configure problem and solver

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Training Batch Size"] = batch_size
e["Problem"]["Testing Batch Size"] = len(testingImages)
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Termination Criteria"]["Max Generations"] = 1

learningRate = 0.001
epochs = 1

# Configure neural network

configure_autencoder(e, img_width, img_height, input_channels, latent_dims)

# Set auxilary settings

e["Console Output"]["Verbosity"] = "Silent"
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = False
e["File Output"]["Path"] = "_korali_result"

# Print Configuration

print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImages)))
print("[Korali] Batch Size: " + str(batch_size))
print("[Korali] Epochs: " + str(epochs))
print("[Korali] Initial Learning Rate: " + str(learningRate))

# Run SGD loop
for epoch in range(epochs):
    tp_start = time.time()
    for step in range(stepsPerEpoch):

        # Creating minibatch
        miniBatchInput = trainingImages[step * batch_size : (step+1) * batch_size] # N x T x C
        miniBatchSolution = [ x[0] for x in miniBatchInput ] # N x C

        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = miniBatchInput
        e["Problem"]["Solution"]["Data"] = miniBatchSolution

        # Reconfiguring solver
        e["Solver"]["Learning Rate"] = learningRate
        e["Solver"]["Termination Criteria"]["Max Generations"] = e["Solver"]["Termination Criteria"]["Max Generations"] + 1

        # Running step
        k.run(e)

    tp_stop = time.time()
    print(f"Epoch {epoch} took {tp_stop-tp_start}")

    # Printing Information
    print("[Korali] --------------------------------------------------")
    print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
    print("[Korali] Learning Rate: " + str(learningRate))
    print('[Korali] Current Training Loss: ' + str(e["Solver"]["Current Loss"]))

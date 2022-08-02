#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
from mnist import MNIST
# from mpi4py import MPI
from statistics import mean
from korali.auxiliar.printing import *
from korali.plot.__main__ import main
import argparse
import time
# from korali.auxiliar.printing import print_args, print_header
parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--epochs',
    help='Maximum Number of epochs to run',
    default=200,
    type=int,
    required=False)
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=0.0001,
    required=False)
parser.add_argument(
    '--learningRateType',
    help='Learning rate type for the selected optimizer',
    default="Decay",
    required=False)
parser.add_argument(
    '--learningRateDecay',
    help='Learning rate decay factor for the selected optimizer',
    default=10,
    required=False)
parser.add_argument(
    '--learningRateMin',
    help='Learning rate decay factor for the selected optimizer',
    default=0.00001,
    required=False)
parser.add_argument(
    '--trainingBatchSize',
    help='Batch size to use for training data',
    default=60,
    required=False)
parser.add_argument(
    '--regularizer',
    help='Reguilrizer type',
    type=str,
    required=False)
parser.add_argument(
    '--regularizerCoefficient',
    help='Reguilrizer Coefficient',
    default=1e-2,
    type=float,
    required=False)
parser.add_argument(
    '--validationBatchSize',
    help='Batch size to use for validation data',
    default=30,
    required=False)
parser.add_argument(
    '--testBatchSize',
    help='Batch size to use for test data',
    default=10,
    required=False)
parser.add_argument(
    "-p",
    "--plot",
    help="Indicates if to plot the losses.",
    required=False,
    action="store_true"
)
parser.add_argument(
    "-s",
    "--save",
    help="Indicates if to save models to _korali_results.",
    required=False,
    action="store_true"
)
parser.add_argument(
    "-t",
    "--test",
    help="Run with reduced number of samples and epochs.",
    required=False,
    action="store_true"
)
parser.add_argument(
    "-l",
    "--load-model",
    help="Load previous model",
    required=False,
    action="store_true"
)


args = parser.parse_args()
k = korali.Engine()
e = korali.Experiment()

### Hyperparameters
MAX_RGB = 255.0
### Lading data ==============================================================================
###Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]
# Data ==================================================================================
t0 = time.time_ns()
mndata = MNIST("./_data")
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()
loading_time = (time.time_ns()-t0) / (float) (10 ** 9) # convert to floating-point second
### Lading data ==============================================================================
args.dataLoadTime = f"{loading_time} s"
img_size = len(trainingImages[0])
args.imgSize = img_size
# Normalize Training Images
trainingImages = [[p/MAX_RGB for p in img] for img in trainingImages]
testingImages = [[p/MAX_RGB for p in img] for img in testingImages]
# ==================================================================================
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
sys.argv = tmp
if args.test:
    args.epochs = 30
    nb_training_samples = 260
    args.trainingBatchSize = 50
    trainingImages = trainingImages[:nb_training_samples]
    args.validationSplit = 60
### Converting images to Korali form (requires a time dimension)
flattendTrainingSamples = [[img] for img in trainingImages]
flattendTrainingTargets = trainingImages

k["Conduit"]["Type"] = "Sequential"
# e["File Output"]["Path"] = "resultsPath"
# Loading previous results, if they exist.
# found = e.loadState()

### Configuring general problem settings
e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Problem"]["Input"]["Data"] = flattendTrainingSamples
e["Problem"]["Solution"]["Data"] = flattendTrainingTargets
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size
### Using a neural network solver (deep learning) for inference
e["Solver"]["Data"]["Validation"]["Split"] = args.validationSplit
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
e["Solver"]["Learning Rate"] = args.learningRate

if args.learningRateType == "Decay":
    e["Solver"]["Learning Rate Type"] = args.learningRateType
    e["Solver"]["Learning Rate Decay Factor"] = args.learningRateDecay
    e["Solver"]["Learning Rate Min"] = args.learningRateMin
    e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Mode"] = "Training"
e["Solver"]["Loss"]["Type"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# MODEL DEFINTION ================================================================================
### Defining the shape of the neural network [autoencoder version of LeNet-1 - http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf (fig. 2)]
## Convolutional Layer with tanh activation function [1x28x28] -> [6x24x24]
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 4 * 24 * 24

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

## Pooling Layer [4x24x24] -> [4x12x12]
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Height"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Width"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 4 * 12 * 12

## Convolutional Layer with tanh activation function [4x12x12] -> [12x8x8]
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = 12 * 8 * 8

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/Tanh"

## Pooling Layer [12x8x8] -> [12x4x4]
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"] = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Height"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Width"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"] = 12 * 4 * 4

## Convolutional Fully Connected Latent Representation Layer [12x4x4] -> [10x1x1]
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"] = 10 * 1 * 1

## Deconvolutional of Fully Connected Latent Representation Layer [10x1x1] -> [12x4x4]
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Output Channels"] = 12 * 4 * 4

## Deonvolutional of Pooling Layer [12x4x4] -> [12x8x8]
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Height"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Width"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Output Channels"] = 12 * 8 * 8

## Deconvolutional of Convolutional Layer [12x8x8] -> [4x12x12]
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Height"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Width"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Output Channels"] = 4 * 12 * 12

## Deconvolutional of Pooling Layer [4x12x12] -> [4x24x24]
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Height"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Width"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Output Channels"] = 4 * 24 * 24

## Deconvolutional of Convolutional Layer [6x28x28] -> [1x28x28]
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Image Height"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Image Width"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Output Channels"] = 1 * 28 * 28
# ================================================================================
### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = args.save
e["Save"]["Problem"] = False
e["Save"]["Solver"] = False
if(args.load_model):
    args.validationSplit = 0.0
    isStateFound = e.loadState("_korali_result/latest")
    if(isStateFound):
        print("[Script] Evaluating previous run...\n")
print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)
# print_args(vars(args), sep=' ', header_width=140)
k.run(e)
# #Evaluating testing set
# e["Problem"]["Input"]["Data"] = flattendTestingSamples
# e["Solver"]["Mode"] = "Testing"
# k.run(e)

# img = e["Solver"]["Evaluation"][0]
# pixels = first_image.reshape((28, 28))
# images_real = [np.array(img[0]).reshape((28,28)) for img in flattendTestingSamples]  # N x C
# images = [np.array(img).reshape((28, 28))  for img in e["Solver"]["Evaluation"]]

# f, axarr = plt.subplots(nb_testing_samples, 2)
# for i, (img, img_real) in enumerate(zip(images, images_real)):
#     plt.imshow(img, cmap='gray')
#     axarr[i,0].imshow(img_real, cmap='gray')
#     axarr[i,1].imshow(img, cmap='gray')

# plt.show()

# k.run(e)
# Getting MSE loss for testing set
# mse = 0.0
# for i, yhat in enumerate(e["Solver"]["Evaluation"]):
#   y = flattendTestingSamples[i][0]
#   mse.append(mean([(img-imghat)*(img-imghat) for (img, imghat) in zip(y, yhat)]))
# for i, yhat in enumerate(e["Solver"]["Evaluation"]):
#     y = flattendTestingSamples[i][0]
#     mse += sum([(px - pxhat) * (px - pxhat) for (px, pxhat) in zip(y, yhat)])

# mse = mse / (nb_testing_samples * 2)
# print("[Korali] Current Testing Loss:  " + str(mse))

# Adjusting learning rate via decay

# for i, yhat in enumerate(e["Solver"]["Evaluation"]):
# ### Plotting Results
if (args.plot):
    SAVE_PLOT = False
    main("_korali_result", False, SAVE_PLOT, False, ["--yscale", "log"])

print_header(width=140)

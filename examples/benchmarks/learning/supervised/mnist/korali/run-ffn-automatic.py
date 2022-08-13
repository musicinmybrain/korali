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
# from mpi4py import MPI
from korali.plot.__main__ import main
import argparse
from autoencoder import configure_autencoder
parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode',
    help='Mode to use, train, predict or all',
    default='all',
    required=False)
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--learningRateType',
    help='Learning rate type for the selected optimizer',
    default="Const",
    required=False)
parser.add_argument(
    "-s",
    "--save",
    help="Indicates if to save models to _korali_results.",
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

add_time_dimension = lambda l : [ [y] for y in l]

args = parser.parse_args()
k = korali.Engine()
e = korali.Experiment()

### Hyperparameters
MAX_RGB = 255.0
### Lading data ==============================================================================
###Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]
### Total 60 000 training samples
# Data ==================================================================================
t0 = time.time_ns()
mndata = MNIST("./_data")
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()
loading_time = (time.time_ns()-t0) / (float) (10 ** 9) # convert to floating-point second
### Lading data ==============================================================================
args.dataLoadTime = f"{loading_time} s"
img_width = 28
img_height = 28
img_size = len(trainingImages[0])
input_channels = 1
assert img_width*img_height == img_size
args.img_width = img_width
args.img_height = img_height
args.img_size = img_height*img_width
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
    args.trainingSetSize = 260
    args.trainingBatchSize = 50
    args.validationSplit = 60

if args.trainingSetSize != "all":
    nb_training_samples = int(args.trainingSetSize)
    trainingImages = trainingImages[:nb_training_samples]

### Converting images to Korali form (requires a time dimension)
trainingTargets = trainingImages
trainingImages = add_time_dimension(trainingImages)
testingImages = add_time_dimension(testingImages)

### Print Header
print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)
### Load Previous model if desired
isStateFound = False
if args.load_model:
    args.validationSplit = 0.0
    isStateFound = e.loadState("_korali_result/latest")
    if not isStateFound:
        sys.exit("No model file for _korali_result/latest found")
    if(isStateFound):
        print("[Script] Evaluating previous run...\n")

k["Conduit"]["Type"] = "Sequential"

### Configuring general problem settings
e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Problem"]["Input"]["Data"] = trainingImages
e["Problem"]["Solution"]["Data"] = trainingTargets
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Testing Batch Size"] = len(testingImages)
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size
### Using a neural network solver (deep learning) for inference
e["Solver"]["Data"]["Validation"]["Split"] = args.validationSplit
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs

e["Solver"]["Learning Rate"] = args.initialLearningRate
e["Solver"]["Learning Rate Type"] = args.learningRateType
e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# MODEL DEFINTION ================================================================================
configure_autencoder(e, img_width, img_height, input_channels, args.latent_dim)
# ================================================================================
### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = args.save
e["File Output"]["Path"] = "_korali_result_pytorch"
e["Save"]["Problem"] = False
e["Save"]["Solver"] = False

#  Training ==================================================================
if args.mode in ["all", "train"]:
    e["Solver"]["Mode"] = "Training"
    k.run(e)
# PREDICTING ================================================================================
e["File Output"]["Enabled"] = False
if args.mode in ["all", "test"]:
    if args.mode == "test" and not isStateFound:
        sys.exit("Cannot predict without loading or training a model.")

    testImage = trainingImages[:args.testBatchSize]
    e["Problem"]["Input"]["Data"] = testingImages
    e["Solver"]["Mode"] = "Predict"
    k.run(e)

# Plotting Results
if (args.plot):
    SAVE_PLOT = False
# Plotting      ===========================================================================
if args.plot:
    results = list(zip(e["Problem"]["Solution"]["Data"], e["Solver"]["Evaluation"]))
    SAMPLES_TO_DISPLAY = 4
    arr_to_img = lambda img : np.reshape(img, (img_height, img_width))
    fig, axes = plt.subplots(nrows=SAMPLES_TO_DISPLAY, ncols=2)
    for ax in axes.flatten():
        y, yhat = random.choice(results)
        ax.imshow(arr_to_img(y), cmap='gist_gray')
        ax.imshow(arr_to_img(yhat), cmap='gist_gray')
    SAVE_PLOT = "None"
    main("_korali_result_pytorch", False, SAVE_PLOT, False, ["--yscale", "linear"])
    plt.show()
print_header(width=140)


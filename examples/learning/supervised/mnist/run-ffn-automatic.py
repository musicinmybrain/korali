#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
from korali.auxiliar.printing import *
sys.path.append(os.path.abspath('./_models'))
sys.path.append(os.path.abspath('..'))
from mnist import MNIST
# from mpi4py import MPI
from korali.plot.__main__ import main
# from cnn_autoencoder import configure_autencoder as autoencoder
from linear_autoencoder import configure_autencoder as autoencoder
from utilities import make_parser
parser = make_parser()
parser.add_argument(
    '--validationBS',
    help='Batch Size to use for the validation set',
    default=256,
    type=int,
    required=False)
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
# Calculate number of samples that is fitting to the BS =====================
nb_training_samples = len(trainingImages)
nb_training_samples = int((nb_training_samples*(1-args.validationSplit))/args.trainingBS)*args.trainingBS
print(f'{nb_training_samples} training samples')
print(f'Discarding {int(len(trainingImages)*(1-args.validationSplit)-nb_training_samples)} training samples')
nb_validation_samples = int((len(trainingImages)*args.validationSplit)/args.testingBS)*args.testingBS
print(f'{nb_validation_samples} validation samples')
print(f'Discarding {int(len(trainingImages)*args.validationSplit-nb_validation_samples)} validation samples')
# nb_training_samples = 256*3
# nb_validation_samples = 256*1
trainingImages = trainingImages[:(nb_training_samples+nb_validation_samples)]
trainingImages = [[p/MAX_RGB for p in img] for img in trainingImages]
testingImages = [[p/MAX_RGB for p in img] for img in testingImages]
### Converting images to Korali form (requires a time dimension)
# Split train data into validation and train data ==============================================
validationImages = trainingImages[:nb_training_samples]
trainingImages = trainingImages[nb_validation_samples:]
nb_training_samples = len(trainingImages)
assert len(validationImages) % args.testingBS == 0
assert len(trainingImages) % args.trainingBS == 0
# ==================================================================================
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
sys.argv = tmp

### Print Header
print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)
### Load Previous model if desired
isStateFound = False
if args.load_model:
    args.validationSplit = 0.0
    isStateFound = e.loadState("_korali_result_automatic/latest")
    if not isStateFound:
        sys.exit("No model file for _korali_result/latest found")
    if(isStateFound):
        print("[Script] Evaluating previous run...\n")

k["Conduit"]["Type"] = "Sequential"
### Configuring general problem settings
e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Problem"]["Input"]["Data"] = add_time_dimension(trainingImages)
e["Problem"]["Solution"]["Data"] = trainingImages
e["Problem"]["Training Batch Size"] = args.trainingBS
e["Problem"]["Testing Batch Sizes"] = [1, args.testingBS]
e["Problem"]["Testing Batch Size"] = args.testingBS
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size
### Using a neural network solver (deep learning) for inference
e["Problem"]["Data"]["Validation"]["Input"] = add_time_dimension(validationImages)
e["Problem"]["Data"]["Validation"]["Solution"] = validationImages
e["Problem"]["Validation Batch Size"] = args.validationBS
# e["Solver"]["Data"]["Validation"]["Split"] = args.validationSplit
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs

e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Learning Rate Type"] = args.learningRateType
e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# MODEL DEFINTION ================================================================================
autoencoder(e, img_width, img_height, input_channels, args.latentDim)
# ================================================================================
### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
e["Random Seed"] = 0xC0FFEE
if args.save:
    e["File Output"]["Enabled"] = True
else:
    e["File Output"]["Enabled"] = False
e["File Output"]["Frequency"] = 0
e["File Output"]["Path"] = "_korali_result_automatic"
e["Save"]["Problem"] = False
e["Save"]["Solver"] = False
e["Save Only"] = "Results"

#  Training ==================================================================
if args.mode in ["all"]:
    e["Solver"]["Mode"] = "Automatic Training"
    k.run(e)

# # PREDICTING ================================================================================
# e["File Output"]["Enabled"] = False
# if args.mode in ["all", "predict"]:
#     if args.mode == "test" and not isStateFound:
#         sys.exit("Cannot predict without loading or training a model.")
#     testImage = trainingImages[:args.testingBS]
#     e["Problem"]["Input"]["Data"] = add_time_dimension(testImage)
#     e["Problem"]["Solution"]["Data"] = testImage
#     e["Solver"]["Mode"] = "Testing"
#     k.run(e)

# # Plotting Results
# if (args.plot):
    # SAMPLES_TO_DISPLAY = 8
    # arr_to_img = lambda img : np.reshape(img, (img_height, img_width))
    # fig, axes = plt.subplots(nrows=SAMPLES_TO_DISPLAY, ncols=2)
    # random.shuffle(testingImages)
    # e["Problem"]["Testing Batch Size"] = args.testingBS
    # e["Solver"]["Mode"] = "Predict"
    # y = [random.choice(testingImages) for i in range(args.testingBS)]
    # e["Problem"]["Input"]["Data"] = add_time_dimension(y)
    # k.run(e)
    # yhat = e["Solver"]["Evaluation"]
    # for y, yhat, ax in list(zip(y[:SAMPLES_TO_DISPLAY], yhat[:SAMPLES_TO_DISPLAY], axes)):
    #     ax[0].imshow(arr_to_img(y), cmap='gist_gray')
    #     ax[1].imshow(arr_to_img(yhat), cmap='gist_gray')
    # SAVE_PLOT = "None"
    # main("_korali_result_automatic", False, SAVE_PLOT, False, ["--yscale", "linear"])
    # plt.show()
# if (args.plot):
#     SAVE_PLOT = False
# # Plotting      ===========================================================================
# if args.plot:
#     results = list(zip(e["Problem"]["Solution"]["Data"], e["Solver"]["Evaluation"]))
#     SAMPLES_TO_DISPLAY = 4
#     arr_to_img = lambda img : np.reshape(img, (img_height, img_width))
#     fig, axes = plt.subplots(nrows=SAMPLES_TO_DISPLAY, ncols=2)
#     for ax in axes.flatten():
#         y, yhat = random.choice(results)
#         ax.imshow(arr_to_img(y), cmap='gist_gray')
#         ax.imshow(arr_to_img(yhat), cmap='gist_gray')
#     SAVE_PLOT = "None"
#     main("_korali_result_pytorch", False, SAVE_PLOT, False, ["--yscale", "linear"])
#     plt.show()
# print_header(width=140)

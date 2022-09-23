#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
import json
from korali.auxiliar.printing import *
sys.path.append(os.path.abspath('./_models'))
sys.path.append(os.path.abspath('..'))
from mnist import MNIST
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from korali.plot.helpers import hlsColors, drawMulticoloredLine
import seaborn as sns
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--testingBS',
    help='Number of examples to resample',
    default=256,
    type=int,
    required=False)
parser.add_argument(
    "--latentDim",
    help="Latent dimension of the encoder",
    default=4,
    required=False,
    type=int
)
parser.add_argument(
    "--reduction-factor",
    help="Factor by which the image width/height gets divided.",
    default=2,
    required=False,
    type=int
)
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
    help="Indicates if to save the results to _results.",
    required=False,
    action="store_true"
)
parser.add_argument(
    "--seed",
    help="Indicates if to save the results to _results.",
    required=False,
    type=int,
    default=42
)
parser.add_argument(
    "--model",
    help="Indicates which model to use.",
    required=False,
    type=str,
    default="linear"
)
parser.add_argument(
    "--yscale",
    help="yscale to plot",
    default="log",
    required=False,
    choices=["linear", "log"]
)
parser.add_argument(
    "--verbosity",
    help="How much output to print.",
    default="Normal",
    required=False,
    choices=["Silent", "Normal", "Detailed"]
)
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    choices=["OneDNN", "CuDNN", "Korali"],
    required=False)

palette = sns.color_palette("deep")
train_c = palette[1]
val_c = palette[3]
lr_c = palette[5]
test_c = palette[4]
f_c = palette[0]
# plt.rcParams['text.usetex'] = True

add_time_dimension = lambda l : [ [y] for y in l]
# ==================================================================================
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp
k = korali.Engine()
e = korali.Experiment()
### Hyperparameters
MAX_RGB = 255.0
### Lading data ==============================================================================
###Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]
### Total 60 000 training samples
### Lading data ==============================================================================
t0 = time.time_ns()
mndata = MNIST("../../../../data/mnist")
mndata.gz = True
testingImages, _ = mndata.load_testing()
loading_time = (time.time_ns()-t0) / (float) (10 ** 9) # convert to floating-point second
args.dataLoadTime = f"{loading_time} s"
### Get dimensions and sizes
img_width = img_height = 28
img_size = len(testingImages[0])
input_channels = 1
assert img_width*img_height == img_size
args.img_width = img_width
args.img_height = img_height
args.img_size = img_height*img_width
### Normalize, shuffel and split data ========================================================
testingImages = [[p/MAX_RGB for p in img] for img in testingImages]
### Print Header
if args.verbosity in ["Normal", "Detailed"]:
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)
### Load Previous model if desired
results_dir = os.path.join("_korali_result")
results_file = os.path.join(results_dir, "latest")
isStateFound = False
k["Conduit"]["Type"] = "Sequential"
### Configuring general problem settings
e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Problem"]["Testing Batch Size"] = args.testingBS
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Mode"] = "Predict"
input_size = output_size = img_width*img_height
img_height_red = int(img_height/args.reduction_factor)
img_width_red = int(img_width/args.reduction_factor)
# ===================== Input Layer
e["Problem"]["Input"]["Size"] = input_size
# ===================== Down Sampling
lidx = 0
e["Solver"]["Neural Network"]["Output Layer"]["Resampling Type"] = "Linear"
e["Solver"]["Neural Network"]["Output Layer"]["Type"] = "Layer/Resampling"
e["Solver"]["Neural Network"]["Output Layer"]["Image Width"] = img_width
e["Solver"]["Neural Network"]["Output Layer"]["Image Height"] = img_height
e["Solver"]["Neural Network"]["Output Layer"]["Output Width"] = img_width_red
e["Solver"]["Neural Network"]["Output Layer"]["Output Height"] = img_height_red
e["Solver"]["Neural Network"]["Output Layer"]["Output Channels"] = img_height_red*img_width_red
# # ===================== Encoder
# e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Linear"
# e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = args.latentDim
# ## Activation ========================
# e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Activation"
# e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Elementwise/ReLU"
# ##  =================== Decoder
# e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Linear"
# e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = img_height_red*img_width_red
# ===================== Up Sampling
# e["Solver"]["Neural Network"]["Output Layer"]["Resampling Type"] = "Linear"
# e["Solver"]["Neural Network"]["Output Layer"]["Type"] = "Layer/Resampling"
# e["Solver"]["Neural Network"]["Output Layer"]["Image Width"] = img_width_red
# e["Solver"]["Neural Network"]["Output Layer"]["Image Height"] = img_height_red
# e["Solver"]["Neural Network"]["Output Layer"]["Output Width"] = img_width
# e["Solver"]["Neural Network"]["Output Layer"]["Output Height"] = img_height
# e["Solver"]["Neural Network"]["Output Layer"]["Output Channels"] = img_width*img_height
# ================================================================================
### Configuring output
e["Console Output"]["Verbosity"] = args.verbosity
e["Random Seed"] = 0xC0FFEE
if args.save:
    e["File Output"]["Enabled"] = True
else:
    e["File Output"]["Enabled"] = False
e["File Output"]["Frequency"] = 0
e["File Output"]["Path"] = results_dir
e["Save Only"] = ["Results", "Solver"]

# PREDICTING ==================================================================================
y = testingImages[:args.testingBS]
e["Problem"]["Input"]["Data"] = add_time_dimension(y)
k.run(e)
yhat= e["Solver"]["Evaluation"]
# Plotting ====================================================================================
args.plot= True
if args.plot:
    #  Plot Reconstruced Images ==================================================
    arr_to_img = lambda img, img_height, img_width : np.reshape(img, (img_height, img_width))
    fig, axes = plt.subplots(nrows=args.testingBS, ncols=3)
    for y, yhat, ax in list(zip(y[:args.testingBS], yhat[:args.testingBS], axes)):
        ax[0].imshow(arr_to_img(y, img_width, img_height), cmap='gist_gray')
        ax[1].imshow(arr_to_img(yhat, img_width_red, img_height_red), cmap='gist_gray')
    # fig.tight_layout()
    if args.save:
        plt.savefig(os.path.join(results_dir, "reduced.png"))
    #  Plot Losses ===============================================================
    sns.set()
    plt.show()

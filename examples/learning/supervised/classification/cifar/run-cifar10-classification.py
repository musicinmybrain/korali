#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import korali
import random
import json
import argparse
# from numba import njit
sys.path.append(os.path.abspath('./_models'))
# Helper Functions ====================================================
sys.path.append(os.path.abspath('.'))
from data_loader import load_CIFAR10
from korali.auxiliar.printing import print_header, print_args, bcolors
# Preprocessing =======================================================
from sklearn.preprocessing import OneHotEncoder
# Plotting ============================================================
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import seaborn as sns
from korali.plot.__main__ import main
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--epochs',
    help='Maximum Number of epochs to run',
    default=30,
    type=int,
    required=False)
parser.add_argument(
    '--validationSplit',
    help='Batch size to use for validation data',
    type=float,
    default=0.01,
    required=False)
parser.add_argument(
    '--trainingBS',
    help='Batch size to use for training data',
    default=128,
    type=int,
    required=False)
parser.add_argument(
    '--testingBS',
    help='Batch size to use for test data',
    default=64,
    type=int,
    required=False)
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
parser.add_argument(
    '--regularizerCoefficient',
    help='Reguilrizer Coefficient',
    default=0,
    type=float,
    required=False)
parser.add_argument(
    "--plot",
    "-p",
    help="Indicates if to plot the losses.",
    required=False,
    action="store_true"
)
parser.add_argument(
    "--save",
    "-s",
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
    "--shuffle",
    help="Indicates whether to shuffel the training, validation and test data.",
    required=False,
    type=bool,
    default=True
)
parser.add_argument(
    "--model",
    help="Indicates which model to use.",
    required=False,
    type=str,
    choices=["logistic", "lenet5"],
    default="logistic"
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
    "--test",
    help="Use a reduce data set for faster training/evaluation.",
    required=False,
    action="store_true"
)
parser.add_argument(
    '--validationBS',
    help='Batch Size to use for the validation set',
    default=32,
    type=int,
    required=False)
parser.add_argument(
    '--mode',
    help='Mode to use, train, predict or all',
    default='Automatic',
    choices=["Training", "Automatic", "Predict", "Plot", "Testing"],
    required=False)
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--learningRateType',
    help='Learning rate type for the selected optimizer',
    choices=["Const", "Step Based", "Time Based"],
    default="Const",
    required=False)
parser.add_argument(
        '-lr',
        '--learningRate',
        help='Learning rate for the selected optimizer',
        default=0.001,
        type=float,
        required=False)
parser.add_argument(
    "-l",
    "--load-model",
    help="Load previous model",
    required=False,
    action="store_true"
)
parser.add_argument(
    "--other",
    help="Can be used to add a folder to distinguish the model inside the results file",
    required=False,
    default="",
    type=str,
)
# parser.add_argument(
#     "--normalize",
#     help="Whether to normalize the images to [0,1].",
#     required=False,
#     action="store_true"
# )
palette = sns.color_palette("deep")
train_c = palette[1]
val_c = palette[3]
lr_c = palette[5]
test_c = palette[4]
f_c = palette[0]
# plt.rcParams['text.usetex'] = True
add_dimension_to_elements = lambda l : [ [y] for y in l]

def rgb2gray(data):
    results = []
    assert len(data[0]) % 3 == 0
    images = int(len(data[0]) / 3)
    for img in data:
        # 32x32x3 => [32, 32, 3]
        gray_scale_img = np.mean(img.reshape(3, 32, 32).transpose(1, 2, 0), axis=2)
        results.append(gray_scale_img.flatten().tolist())
    return results

# ==================================================================================
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp
#  MODEL SELECTION ===========================================================
if args.model == "logistic":
    from linear_clf import LogisticRegression as classifier
elif args.model == "lenet5":
    from lenet import LeNet5 as classifier
else:
    sys.exit(f"{args.model} is not a valid model.")
# Load Korali Engine =========================================================
k = korali.Engine()
k["Conduit"]["Type"] = "Sequential"
# Loading The DATA ======================================================================================
### Loading CIFAR data [32x32 images with {0,..,9} as label - https://www.cs.toronto.edu/~kriz/cifar.html
### 5x 10000 training samples in batch files and one test file
t0 = time.time_ns()
X_train, y_train, X_test, y_test = load_CIFAR10("../../../../../data/cifar10")
args.dataLoadTime = f"{(time.time_ns()-t0) / (float) (10 ** 9)} s"
### One hot encode the labels
t0 = time.time_ns()
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(add_dimension_to_elements(y_train))
y_test = onehot_encoder.transform(add_dimension_to_elements(y_test))
args.oneHotEncodingTime = f"{(time.time_ns()-t0) / (float) (10 ** 9)} s"
### Get Dimensions and Sizes =================================================================
img_width = img_height = 32
label_size = len(y_train[0])
input_channels = 1
classes = 10
args.img_width = img_width
args.img_height = img_height
args.img_size = img_height*img_width
args.label_size = label_size
# Calculate number of samples that is dividable by the BS and discard some otherwise
nb_training_samples = X_train.shape[0]
nb_training_samples = nb_training_samples*(1-args.validationSplit)
nb_training_batches = nb_training_samples/args.trainingBS
nb_training_samples = int(nb_training_batches)*args.trainingBS
nb_testing_samples = X_test.shape[0]
if args.test:
    nb_training_samples = args.trainingBS*10
nb_validation_samples = int((X_train.shape[0]*args.validationSplit)/args.validationBS)*args.validationBS
if args.test:
    nb_validation_samples = args.validationBS*1
X_train = X_train[:(nb_training_samples+nb_validation_samples)]
X_train = X_train[:(nb_training_samples+nb_validation_samples)]
### Normalize ================================================================================
# if args.normalize:
MAX_RGB = 255.0
t0 = time.time_ns()
# trainingSet = [([p/MAX_RGB for p in img], label) for img, label in trainingSet]
X_train = X_train/MAX_RGB
X_test = X_test/MAX_RGB
args.normalizationTime = f"{(time.time_ns()-t0) / (float) (10 ** 9)} s"
# Shuffle the data ===========================================================================
indices = np.arange(X_train.shape[0])
random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]
# Convert to grayscale image =================================================================
X_train = rgb2gray(X_train)
# TODO test data
# X_test = rgb2gray(X_train)
img_size = len(X_train[0])
# Split data in train and validation set =====================================================
X_train = X_train[:nb_training_samples]
y_train = y_train[:nb_training_samples]
X_val = X_train[-nb_validation_samples:]
y_val = y_train[-nb_validation_samples:]
nb_training_samples = len(X_train)
assert len(X_train) % args.trainingBS == 0
assert len(X_val) % args.validationBS == 0
assert img_width*img_height*input_channels == img_size
### Print Header
if args.verbosity in ("Normal", "Detailed"):
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)
    print(f'{nb_training_samples} training samples')
    print(f'Discarding {(nb_training_samples % args.trainingBS)*args.trainingBS} training samples')
    print(f'{nb_validation_samples} validation samples')
    print(f'Discarding {int(len(X_train)*args.validationSplit-nb_validation_samples)} validation samples')

# Convert to list for KORALI =================================================================
y_train = y_train.tolist()
y_val = y_val.tolist()

# ==============================================================================================
# set up the experiments for different optimizerers
# ==============================================================================================
eList = []
optimizers = [
    "AdaBelief",
    "AdaGrad",
    "Adam",
    "MadGrad",
    "Momentum",
    "RMSProp",
    "SGD"
]
current_optimizers = ["Adam"]
# current_optimizers = optimizers
for i, optimizer in enumerate(current_optimizers):
    e = korali.Experiment()
    results_dir = os.path.join("_korali_results", args.model, args.other, optimizer)
    results_file = os.path.join(results_dir, "latest")
    ### Loading previous models if desired
    if args.load_model:
        e.loadState(results_file)
    # Experiment Setting ===========================================================
    e["Random Seed"] = 0xC0FFEE
    e["Console Output"]["Verbosity"] = args.verbosity
    # Solver Settings  =============================================================
    e["Solver"]["Type"] = "Learner/DeepSupervisor"
    e["Solver"]["Mode"] = "Automatic Training"
    e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
    e["Solver"]["Loss Function"] = "Cross Entropy"
    e["Solver"]["Metrics"]["Type"] = "Accuracy"
    e["Solver"]["Learning Rate"] = args.learningRate
    e["Solver"]["Neural Network"]["Engine"] = args.engine
    e["Solver"]["Optimizer"]["Type"] = "learner/deepSupervisor/optimizers/f"+optimizer
    e["Solver"]["Data"]["Input"]["Shuffel"] = False
    e["Solver"]["Data"]["Training"]["Shuffel"] = True
    # Problem Settings =============================================================
    e["Problem"]["Type"] = "Supervised Learning"
    e["Problem"]["Training Batch Size"] = args.trainingBS
    e["Problem"]["Testing Batch Size"] = nb_validation_samples
    e["Problem"]["Max Timesteps"] = 1
    # Data
    e["Problem"]["Input"]["Data"] = add_dimension_to_elements(X_train)
    e["Problem"]["Solution"]["Data"] = y_train
    e["Problem"]["Input"]["Size"] = img_size*input_channels
    e["Problem"]["Solution"]["Size"] = label_size
    e["Problem"]["Data"]["Validation"]["Input"] = add_dimension_to_elements(X_val)
    e["Problem"]["Data"]["Validation"]["Solution"] = y_val
    # MODEL DEFINTION ==============================================================
    classifier(e, img_width, img_height, label_size, input_channels)
    # File Settings ================================================================
    e["File Output"]["Enabled"] = args.save
    e["File Output"]["Frequency"] = 0
    e["File Output"]["Path"] = results_dir
    # e["Save Only"] = ["Current Generation" ,"Run ID", "Solver"]
    eList.append(e)
k.run(eList)

# # PREDICTING ==================================================================================
# e["File Output"]["Enabled"] = False
# if args.mode in ["Predict", "Testing"]:
#     if not isStateFound:
#         sys.exit("Cannot predict without loading or training a model.")
#     e["Solver"]["Mode"] = "Testing"
#     e["Problem"]["Input"]["Data"] = add_dimension_to_elements(validationImages)
#     e["Problem"]["Solution"]["Data"] = validationLabels
#     k.run(e)

# # # Plotting Results
if args.plot:
    # Plot Losses ======================================================================
    # Note: need to run run-classification.py with --save flag once to get output files.
    SAVE_PLOT = "None"
    main(results_dir, False, SAVE_PLOT, False, ["--yscale", "linear"])
    plt.show()

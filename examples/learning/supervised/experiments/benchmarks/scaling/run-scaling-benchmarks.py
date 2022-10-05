#!/usr/bin/env ipython
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
from korali.auxiliar.printing import *
from korali.plot.__main__ import main
import math
import copy
import argparse
import seaborn as sns
sys.path.append(os.path.abspath('./_models'))

sns.set()
palette = sns.color_palette("tab10")
train_c = palette[2]
val_c = palette[3]
lr_c = palette[5]
test_c = palette[4]
model_c = palette[1]
f_c = palette[0]
plt.rcParams['text.usetex'] = True

k = korali.Engine()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--epochs',
    help='Maximum Number of epochs to run',
    default=3,
    type=int,
    required=False)
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='SGD',
    required=False)
# Learning Rate ==================================================
parser.add_argument(
    '--initialLearningRate',
    help='Learning rate for the selected optimizer',
    default=0.001,
    type=float,
    required=False)
parser.add_argument(
    '--trainingSetSize',
    help='Batch size to use for training data',
    default=10000,
    required=False)
parser.add_argument(
    '--trainingBS',
    help='Batch size to use for training data',
    type=int,
    default=32,
    choices=[32, 1024],
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
parser.add_argument(
    "-m",
    "--model",
    help="Run model with one hidden or many hidden layers (of same size).",
    type=str,
    required=True,
    choices=["single", "multi"]
)
parser.add_argument(
    "--other",
    help="Can be used to add a folder to distinguish the model inside the results file",
    required=False,
    default="",
    type=str,
)
parser.add_argument(
    "--weight-dim",
    help="Weights Siz",
    required=True,
    choices=[15,17,19,21,23,25],
    type=int,
)
parser.add_argument(
    '--verbosity',
    help='Verbosity to print',
    default="Normal",
    required=False)

np.random.seed(0xC0FFEE)
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp

if args.verbosity in ["Normal", "Detailed"]:
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)

e = korali.Experiment()
e["Random Seed"] = 0xC0FFEE
#  MODEL SELECTION ===========================================================
SMALLEST_LAYER_SIZE_EXPONENT = 7
SMALLEST_LAYER_SIZE = 2**SMALLEST_LAYER_SIZE_EXPONENT
if args.model == "single":
    from models import single_layer_ffnn as nn
    layers = 1
    assert (args.weight_dim-1) % 2 == 0
    exponent = ((args.weight_dim-1)/2)
    dim=int(2**exponent)
    e["Problem"]["Input"]["Size"] = dim
    e["Problem"]["Solution"]["Size"] = dim
    if args.verbosity in ("Normal", "Detailed"):
        print(f"layers {layers}")
        print(f"layer dimension 2^{exponent}")
elif args.model == "multi":
    from models import multi_layer_ffnn as nn
    OUTPUT_LAYERS = 1
    dim=SMALLEST_LAYER_SIZE
    layers = 2**(args.weight_dim-(2*SMALLEST_LAYER_SIZE_EXPONENT+OUTPUT_LAYERS))
    e["Problem"]["Input"]["Size"] = dim
    e["Problem"]["Solution"]["Size"] = dim
    if args.verbosity in ("Normal", "Detailed"):
        print(f"layer dimension 2^{SMALLEST_LAYER_SIZE_EXPONENT}")
        print(f"layers {layers}")
else:
    sys.exit(f"{args.model} is not a valid model.")
# ============================================================================

path = os.path.join("_korali_result", f"weights_{args.weight_dim}", args.other, args.model)

# Create Training Data =====================================================================
X_train = np.random.rand(args.trainingSetSize, 1, dim).tolist()
y_train = np.random.rand(args.trainingSetSize, dim).tolist()
# ==========================================================================================
### Loading previous models
if args.load_model:
    args.validationSplit = 0.0
    e.loadState(os.path.join(path, "latest"))

def flops_per_layer(bs, ic, oc):
    forward_gemm = (2*ic*oc*bs)/(2*(bs*oc)+ic*(bs+oc)*4)
    backward_gemm = 2*forward_gemm
    forward_acitvation = bs*dim
    backward_acitvation = bs*dim
    return forward_gemm+backward_gemm+forward_acitvation+backward_acitvation

def flops(training_set_size, bs, dim, hidden_layers):
    layers = hidden_layers+1
    iterrations = training_set_size/bs;
    flops = iterrations*layers*flops_per_layer(bs, dim, dim)
    return flops

if args.verbosity in ("Normal", "Detailed"):
    print(f"FLOPS: {flops(args.trainingSetSize, args.trainingBS, dim, layers)}")
e["Problem"]["Description"] = f"FLOPS: {flops(args.trainingSetSize, args.trainingBS, dim, dim)}"
e["Solver"]["Batch Concurrency"] = 1
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = args.trainingBS

e["Problem"]["Input"]["Data"] = X_train
e["Problem"]["Solution"]["Data"] = y_train

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Mode"] = "Automatic Training"
e["Solver"]["Loss Function"] = "Mean Squared Error"

e["Solver"]["Learning Rate"] = args.initialLearningRate

e["Solver"]["Data"]["Input"]["Shuffel"] = False
e["Solver"]["Data"]["Training"]["Shuffel"] = False
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Optimizer"]["Type"] = "learner/deepSupervisor/optimizers/f"+args.optimizer
# Set Model ================================================================================
nn(e, dim, layers, SMALLEST_LAYER_SIZE)
# ==========================================================================================
# e["Console Output"]["Frequency"] = 0
e["Console Output"]["Verbosity"] = args.verbosity
e["File Output"]["Enabled"] = args.save
e["File Output"]["Frequency"] = 0
e["File Output"]["Path"] = path
# e["Save Only"] = ["Current Generation" ,"Run ID", "Solver"]
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
k["Conduit"]["Type"] = "Sequential"
k.run(e)
# Store X_train, y_train, X_val, y_val (might change from input due to validaiton split) ===============
# data = {}
# for i, key in enumerate(current_keys):
#     data[key] = {
#         "X_train": eList[i]["Problem"]["Input"]["Data"],
#         "y_train": eList[i]["Problem"]["Solution"]["Data"],
#         "X_val": eList[i]["Problem"]["Data"]["Validation"]["Input"],
#         "y_val": eList[i]["Problem"]["Data"]["Validation"]["Solution"]
#     }
# # Predict on the unseen test set =======================================================================
# for i, _ in enumerate(eList):
#     # # ### Obtaining inferred results from the NN and comparing them to the actual solution
#     eList[i]["Solver"]["Mode"] = "Testing"
#     eList[i]["Problem"]["Testing Batch Size"] = args.testingSetSize
#     eList[i]["Problem"]["Input"]["Data"] = X_test_k
#     eList[i]["Problem"]["Solution"]["Data"] = y_test_k
#     eList[i]["File Output"]["Enabled"] = False
# k.run(eList)
# # Plotting Training Results
# if (args.plot):
#     regularizer_type = ["","$+ \\lambda||w||^2$"]
#     # Plot ture function, training data, validation data ===================================================
#     fig, axes = plt.subplots(len(current_keys))
#     if len(current_keys) == 1:
#         axes = [axes]
#     for i, key in enumerate(current_keys):
#         axes[i].plot(np.linspace(x_min, x_max, nb_x_values), f(np.linspace(x_min, x_max, nb_x_values)), label="$f(x)$", color = f_c)
#         axes[i].scatter(data[key]["X_train"], data[key]["y_train"], label="$y_{\\textrm{train}}=f(x)+\\epsilon$", color=train_c)
#         axes[i].scatter(data[key]["X_val"], data[key]["y_val"], label="$y_{\\textrm{val}}=f(x)+\\epsilon$", color=val_c)
#         axes[i].scatter(eList[i]["Problem"]["Input"]["Data"], eList[i]["Problem"]["Solution"]["Data"], label="$y_{\\textrm{test}}=f(x)+\\epsilon$", color=test_c)

#     # Obtain predctions of the nn model to plot its function ===============================================
#     for i, _ in enumerate(eList):
#         # # ### Obtaining inferred results from the NN and comparing them to the actual solution
#         eList[i]["Solver"]["Mode"] = "Predict"
#         eList[i]["Problem"]["Testing Batch Size"] = nb_x_values
#         eList[i]["Problem"]["Input"]["Data"] = X_space_k
#         eList[i]["File Output"]["Enabled"] = False
#     k.run(eList)
#     # Calculate the size of the weights ====================================================================
#     yhat = []
#     weight_sum = []
#     for i, _ in enumerate(eList):
#         yhat.append([ y[0] for y in eList[i]["Solver"]["Evaluation"] ])
#         weight_sum.append(sum([ w**2 for w in eList[i]["Solver"]["Hyperparameters"] ]))

#     # Plot the predicted function of the nn model ==========================================================
#     for i, ax in enumerate(axes):
#         if current_keys[i] == keys[0]:
#             coef_size = ""
#             reg_t = regularizer_type[0]
#         else:
#             coef_size = " and $\\lambda = $" + f'{args.regularizerCoefficient:.4f}'
#             reg_t = regularizer_type[1]
#         ax.plot(X_space, yhat[i], label="$\\hat{y}(x)$ "+reg_t+" with $||w||^2 = $ " + f'{weight_sum[i]:.4f}'+coef_size, color=model_c)
#         ax.legend()
#         ax.set_yscale("linear")
#     for key in current_keys:
#         SAVE_PLOT = "None"
#         main(paths[key], False, SAVE_PLOT, False, ["--yscale", "linear"])
#     plt.show()
# print_header(width=140)

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
sns.set()
palette_tab10 = sns.color_palette("tab10", 10)
plt.rcParams['text.usetex'] = True

def set_complex_model(e, input_dims):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param input_dims: encoding dimension
    """
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_dims
    e["Problem"]["Solution"]["Size"] = input_dims
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Output Channels"] = 32


def set_simple_model(e, input_dims):
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_dims
    e["Problem"]["Solution"]["Size"] = input_dims
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

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
    default=30000,
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
    default=0.005,
    required=False)
parser.add_argument(
    '--learningRateType',
    help='Learning rate type for the selected optimizer',
    default="Const",
    required=False)
parser.add_argument(
    '--learningRateDecay',
    help='Learning rate decay factor for the selected optimizer',
    default=10,
    required=False)
parser.add_argument(
    '--trainingSetSize',
    help='Batch size to use for training data',
    default=50,
    required=False)
parser.add_argument(
    '--trainingBatchSize',
    help='Batch size to use for training data',
    default=20,
    required=False)
parser.add_argument(
    '--testBatchSize',
    help='Batch size to use for test data',
    default=400,
    required=False)
parser.add_argument(
    '--regularizerType',
    help='Reguilrizer type',
    default="L2",
    type=str,
    required=False)
parser.add_argument(
    '--regularizerCoefficient',
    help='Reguilrizer Coefficient',
    default=0.05,
    type=float,
    required=False)
parser.add_argument(
    '--validationSplit',
    help='Batch size to use for validation data',
    default=0.1,
    required=False)
parser.add_argument(
    "-n",
    "--noise",
    help="Indicates",
    required=False,
    action="store_true"
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
    "--std",
    help="Standard Deviation",
    required=False,
    default=0.9
)
parser.add_argument(
    "-m",
    "--model",
    help="Comples or simple model to chose.",
    type=str,
    default="simple",
    required=False,
    choices=["simple", "complex"]
)


# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp
# Test functions to use
f = np.vectorize(lambda x: -(1.4 - 3.0 * x) * math.sin(18.0 * x))
x_min = 0
x_max = 1.3

X_train = np.sort(np.random.uniform(x_min, x_max, args.trainingSetSize))
X_test = np.linspace(x_min, x_max, args.testBatchSize)
eps = np.random.normal(loc=0.0, scale=args.std, size=args.trainingSetSize)
y_train = f(X_train)+eps
y_test = f(X_test)
X_train_k = X_train.flatten().tolist()
X_test_k = X_test.flatten().tolist()
y_train_k = y_train.flatten().tolist()
y_test_k = y_test.flatten().tolist()
X_train_k = [ [[x]] for x in X_train_k ]
X_test_k = [ [[x]] for x in X_test_k ]
y_train_k = [ [y] for y in y_train_k ]
y_test_k = [ [y] for y in y_test_k ]


print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)

np.random.seed(0xC0FFEE)

eList = []
for i, path in enumerate(["_korali_result", "_korali_resultL2"]):
    e = korali.Experiment()
    e["Problem"]["Description"] = "Supervised Learning Problem: $y(x)=\\tanh(\\exp(\\sin(\\texttt{trainingInputSet})))\\cdot\\texttt{scaling}$"
    e["Problem"]["Type"] = "Supervised Learning"
    e["Problem"]["Max Timesteps"] = 1
    e["Problem"]["Training Batch Size"] = args.trainingBatchSize
    e["Problem"]["Testing Batch Size"] = args.testBatchSize

    e["Problem"]["Input"]["Data"] = X_train_k
    e["Problem"]["Input"]["Size"] = 1
    e["Problem"]["Solution"]["Data"] = y_train_k
    e["Problem"]["Solution"]["Size"] = 1

    # e["Problem"]["Data"]["Validation"]["Input"] = validationInputSet
    e["Solver"]["Data"]["Validation"]["Split"] = args.validationSplit
    # ### Using a neural network solver (deep learning) for training

    e["Solver"]["Type"] = "Learner/DeepSupervisor"
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Loss"]["Type"] = "Mean Squared Error"

    e["Solver"]["Learning Rate"] = float(args.learningRate)
    if args.learningRateType:
        e["Solver"]["Learning Rate Type"] = args.learningRateType
        e["Solver"]["Learning Rate Decay Factor"] = args.learningRateDecay
        e["Solver"]["Learning Rate Save"] = True

    e["Solver"]["Batch Concurrency"] = 1
    e["Solver"]["Data"]["Training"]["Shuffel"] = True
    e["Solver"]["Neural Network"]["Engine"] = args.engine
    e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

    if args.model == "simple":
        set_simple_model(e, len(X_train_k[0]))
    elif args.model == "complex":
        set_complex_model(e, len(X_train_k[0]))

    e["Console Output"]["Frequency"] = 1
    e["Console Output"]["Verbosity"] = "Normal"
    e["File Output"]["Enabled"] = args.save
    e["File Output"]["Frequency"] = 1 if args.epochs <= 100 else args.epochs/100
    e["File Output"]["Path"] = path
    e["Save"]["Problem"] = False
    e["Save"]["Solver"] = False
    e["Random Seed"] = 0xC0FFEE

    ### Training the neural network
    if(args.load_model):
        args.validationSplit = 0.0
        isStateFound = e.loadState("_korali_result/latest")
        if(isStateFound):
            print("[Script] Evaluating previous run...\n")

    e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
    k["Conduit"]["Type"] = "Sequential"
    if i == 1:
        e["Solver"]["Regularizer"]["Type"] = args.regularizerType
        e["Solver"]["Regularizer"]["Coefficient"] = args.regularizerCoefficient
    eList.append(e)
k.run(eList)

for i in range(2):
    # # ### Obtaining inferred results from the NN and comparing them to the actual solution
    eList[i]["Solver"]["Mode"] = "Predict"
    eList[i]["Problem"]["Input"]["Data"] = X_test_k
    # ### Running Testing and getting results
    eList[i]["File Output"]["Enabled"] = False
k.run(eList)
yhat = []
weight_sum = []
for i in range(2):
    yhat.append([ y[0] for y in eList[i]["Solver"]["Evaluation"] ])
    weight_sum.append(sum([ w**2 for w in eList[i]["Solver"]["Hyperparameters"] ]))

# ### Plotting Results
if (args.plot):
    fig, axes = plt.subplots(2)
    for i, type in enumerate(["","$+ \\lambda||w||^2$"]):
        axes[i].plot(np.linspace(x_min, x_max, 400), f(np.linspace(x_min, x_max, 400)), label="$f(x)$")
        axes[i].scatter(X_train, y_train, label="$f(x)+\\epsilon$", color='orange')
        l = "" if type == "" else " and $\\lambda = $" + f'{args.regularizerCoefficient:.4f}'
        axes[i].plot(X_test, yhat[i], label="$\\hat{y}(x)$ "+type+" with $||w||^2 = $ " + f'{weight_sum[i]:.4f}'+l, color='green')
        axes[i].legend()
        axes[i].set_yscale("linear")
    for p in ["_korali_result", "_korali_resultL2"]:
        SAVE_PLOT = "None"
        main(p, False, SAVE_PLOT, False, ["--yscale", "log"])
    plt.show()

print_header(width=140)

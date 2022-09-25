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
    default=10000,
    type=int,
    required=False)    
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
# Learning Rate ==================================================
parser.add_argument(
    '--initialLearningRate',
    help='Learning rate for the selected optimizer',
    default=0.001,
    type=float,
    required=False)
parser.add_argument(
    '--learningRateType',
    help='Learning rate type for the selected optimizer',
    default="Const",
    required=False)
parser.add_argument(
    '--learningRateDecayFactor',
    help='Learning rate decay factor for the selected optimizer',
    default=10,
    type=float,
    required=False)
parser.add_argument(
    '--learningRateSteps',
    help='Steps until we reduce the learning rate.',
    default=300,
    type=float,
    required=False)
parser.add_argument(
    '--learningRateLowerBound',
    help='Learning rate decay factor for the selected optimizer',
    default=0.00001,
    type=float,
    required=False)
# ================================================================
parser.add_argument(
    '--trainingSetSize',
    help='Batch size to use for training data',
    default=50,
    required=False)
parser.add_argument(
    '--trainingBatchSize',
    help='Batch size to use for training data',
    type=int,
    default=20,
    required=False)
parser.add_argument(
    '--validationSplit',
    help='Batch size to use for validation data',
    default=0.1,
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
    default=0.0
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
#  MODEL SELECTION ===========================================================
if args.model == "simple":
    from simple_model import set_simple_model as regressor
elif args.model == "complex":
    from complex_model import set_complex_model as regressor
else:
    sys.exit(f"{args.model} is not a valid model.")
#  ===========================================================================
add_sample_dimension = add_time_dimension = lambda l : [ [y] for y in l]
# Test functions to use
f = np.vectorize(lambda x: -(1.4 - 3.0 * x) * math.sin(18.0 * x))
x_min = 0
x_max = 1.3

X_train = np.sort(np.random.uniform(x_min, x_max, args.trainingSetSize))
X_test = np.linspace(x_min, x_max, args.testBatchSize)
eps = np.random.normal(loc=0.0, scale=args.std, size=args.trainingSetSize)
y_train = f(X_train)+eps
y_test = f(X_test)

y_train = add_sample_dimension(y_train.tolist())
y_test = add_sample_dimension(y_test.tolist())
X_train = add_sample_dimension(X_train.tolist())
X_test = add_sample_dimension(X_test.tolist())
input_dims = len(X_train[0])

print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)

np.random.seed(0xC0FFEE)

eList = []
# run model with and without regularizer
for i, path in enumerate(["_korali_result", "_korali_resultL2"]):
    e = korali.Experiment()
    e["Problem"]["Type"] = "learning/Supervised Learning"
    e["Problem"]["Description"] = "Supervised Learning Problem: $y(x)=\\tanh(\\exp(\\sin(\\texttt{trainingInputSet})))\\cdot\\texttt{scaling}$"
    e["Problem"]["Max Timesteps"] = 1
    e["Problem"]["Training Batch Size"] = args.trainingBatchSize
    e["Problem"]["Testing Batch Size"] = args.testBatchSize

    e["Problem"]["Input"]["Data"] = add_time_dimension(X_train)
    e["Problem"]["Solution"]["Data"] = y_train
    e["Solver"]["Data"]["Validation"]["Split"] = args.validationSplit

    e["Solver"]["Type"] = "Learner/DeepSupervisor"
    e["Solver"]["Mode"] = "Automatic Training"
    e["Solver"]["Loss Function"] = "Mean Squared Error"

    e["Solver"]["Learning Rate"] = args.initialLearningRate
    if args.learningRateType:
        e["Solver"]["Learning Rate Type"] = args.learningRateType
        e["Solver"]["Learning Rate Decay Factor"] = args.learningRateDecayFactor
        e["Solver"]["Learning Rate Lower Bound"] = args.learningRateLowerBound
        e["Solver"]["Learning Rate Steps"] = args.learningRateSteps
        e["Solver"]["Learning Rate Save"] = True

    e["Solver"]["Batch Concurrency"] = 1
    e["Solver"]["Data"]["Training"]["Shuffel"] = True
    e["Solver"]["Neural Network"]["Engine"] = args.engine
    e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer
    # MODEL DEFINTION ================================================================================
    regressor(e, input_dims)
    # ================================================================================
    e["Console Output"]["Frequency"] = 1
    e["Console Output"]["Verbosity"] = "Normal"
    e["File Output"]["Enabled"] = args.save
    e["File Output"]["Frequency"] = 0
    e["File Output"]["Path"] = path
    e["Save Only"] = ["Current Generation" ,"Run ID", "Results", "Solver"]
    e["Random Seed"] = 0xC0FFEE

    ### Loading previous models
    if(args.load_model):
        args.validationSplit = 0.0
        isStateFound = e.loadState(os.path.join("_korali_result", path))
        if(isStateFound):
            print("[Script] Evaluating previous run...\n")

    e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
    k["Conduit"]["Type"] = "Sequential"
    if i == 1:
        e["Solver"]["Regularizer"]["Type"] = args.regularizerType
        e["Solver"]["Regularizer"]["Coefficient"] = args.regularizerCoefficient
    eList.append(e)
k.run(eList)

# ### Plotting Training Results
if (args.plot):
    fig, axes = plt.subplots(2)
    for i, type in enumerate(["","$+ \\lambda||w||^2$"]):
        axes[i].plot(np.linspace(x_min, x_max, 400), f(np.linspace(x_min, x_max, 400)), label="$f(x)$", color = f_c)
        axes[i].scatter(eList[i]["Problem"]["Input"]["Data"], eList[i]["Problem"]["Solution"]["Data"], label="$y_{\\textrm{train}}=f(x)+\\epsilon$", color=train_c)
        axes[i].scatter(eList[i]["Problem"]["Data"]["Validation"]["Input"], eList[i]["Problem"]["Data"]["Validation"]["Solution"], label="$y_{\\textrm{val}}=f(x)+\\epsilon$", color=val_c)

    for i, _ in enumerate(eList):
        # # ### Obtaining inferred results from the NN and comparing them to the actual solution
        eList[i]["Solver"]["Mode"] = "Predict"
        eList[i]["Problem"]["Input"]["Data"] = add_time_dimension(X_test)
        # ### Running Testing and getting results
        eList[i]["File Output"]["Enabled"] = False
    k.run(eList)

    yhat = []
    weight_sum = []
    for i, _ in enumerate(eList):
        yhat.append([ y[0] for y in eList[i]["Solver"]["Evaluation"] ])
        weight_sum.append(sum([ w**2 for w in eList[i]["Solver"]["Hyperparameters"] ]))

# ### Plotting Testing Results
    for i, type in enumerate(["","$+ \\lambda||w||^2$"]):
    # for i, type in enumerate([""]):
        l = "" if type == "" else " and $\\lambda = $" + f'{args.regularizerCoefficient:.4f}'
        axes[i].plot(X_test, yhat[i], label="$\\hat{y}(x)$ "+type+" with $||w||^2 = $ " + f'{weight_sum[i]:.4f}'+l, color=test_c)
        axes[i].legend()
        axes[i].set_yscale("linear")
    for p in ["_korali_result", "_korali_resultL2"]:
    # for p in ["_korali_result"]:
        SAVE_PLOT = "None"
        main(p, False, SAVE_PLOT, False, ["--yscale", "linear"])
    plt.show()

print_header(width=140)

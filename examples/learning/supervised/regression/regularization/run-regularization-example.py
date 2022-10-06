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
from simple_model import set_simple_model as regressor

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
    default=30000,
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
    default=8*6,
    required=False)
parser.add_argument(
    '--trainingBS',
    help='Batch size to use for training data',
    type=int,
    default=8,
    required=False)
parser.add_argument(
    '--validationSplit',
    help='Batch size to use for validation data',
    default=40,
    required=False)
parser.add_argument(
    '--testingSetSize',
    help='Batch size to use for test data',
    default=40,
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
    default=0.2,
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
    "--load-path",
    help="Load-from different path than save path.",
    default="",
    required=False
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
    default=0.7
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
parser.add_argument(
    '--outliers',
    help='Batch size to use for training data',
    default=100,
    required=False
)
parser.add_argument(
    '--type',
    help='If to run no-regularization, regularization or both',
    default="all",
    choices=["all", "no", "reg"],
    required=False
)
parser.add_argument(
    '--modelFunction',
    help='Model function y=f(x) that we want to learn.',
    default="complex",
    choices=["x2", "complex"],
    required=False
)
parser.add_argument(
    "--other",
    help="Can be used to add a folder to distinguish the model inside the results file",
    required=False,
    default="",
    type=str,
)

np.random.seed(0xC0FFEE)
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp
# Test functions to use
if args.modelFunction == "x2":
    f= np.vectorize(lambda x: x*x)
    x_min = -4
    x_max = 4
else:
    f = np.vectorize(lambda x: -(1.4 - 3.0 * x) * math.sin(18.0 * x))
    x_min = 0
    x_max = 1.3


X_train = np.sort(np.random.uniform(x_min, x_max, args.trainingSetSize))
eps = np.random.normal(loc=0.0, scale=args.std, size=args.trainingSetSize)
y_train = f(X_train)+eps
X_val = np.sort(np.random.uniform(x_min, x_max, args.validationSplit))
eps = np.random.normal(loc=0.0, scale=args.std, size=args.validationSplit)
y_val = f(X_val)+eps
X_test = np.sort(np.random.uniform(x_min, x_max, args.testingSetSize))
eps = np.random.normal(loc=0.0, scale=args.std, size=args.testingSetSize)
y_test = f(X_test)

# linear space of data to get a plot of the function of the model
nb_x_values = 400
X_space = np.linspace(x_min, x_max, nb_x_values)

X_train = X_train.tolist()
X_val = X_val.tolist()
X_space = X_space.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_val = y_val.tolist()
y_test = y_test.tolist()

X_train_k = [ [[x]] for x in X_train ]
X_val_k = [ [[x]] for x in X_val ]
X_test_k = [ [[x]] for x in X_test ]
y_train_k = [ [y] for y in y_train ]
y_val_k = [ [y] for y in y_val ]
y_test_k = [ [y] for y in y_test ]
X_space_k = [ [[x]] for x in X_space ]

# input_dim = len(X_train[0])
print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)


eList = []
keys = ["no", "regularization"]
paths = {keys[0]: os.path.join("_korali_result", args.other, "_korali_result_no_regularization"),
        keys[1]:  os.path.join("_korali_result", args.other, "_korali_result_L2")}
if args.load_path:
    load_paths = {keys[0]: os.path.join("_korali_result", args.load_path, "_korali_result_no_regularization"),
            keys[1]:  os.path.join("_korali_result", args.load_path, "_korali_result_L2")}
else:
    load_paths = paths

current_keys = ["regularization"]
if args.type == "all":
    current_keys = ["no", "regularization"]
elif args.type == "reg":
    current_keys = ["regularization"]
elif args.type == "no":
    current_keys = ["no"]

for i, key in enumerate(current_keys):
    e = korali.Experiment()
    e["Random Seed"] = 0xC0FFEE
    ### Loading previous models
    if args.load_model:
        args.validationSplit = 0.0
        e.loadState(os.path.join(load_paths[key], "latest"))

    e["Problem"]["Description"] = key
    e["Problem"]["Type"] = "Supervised Learning"
    e["Problem"]["Max Timesteps"] = 1
    e["Problem"]["Training Batch Size"] = args.trainingBS
    e["Problem"]["Testing Batch Sizes"] = [nb_x_values,  args.testingSetSize]

    e["Problem"]["Data"]["Validation"]["Input"] = X_val_k
    e["Problem"]["Data"]["Validation"]["Solution"] = y_val_k
    e["Problem"]["Input"]["Data"] = X_train_k
    e["Problem"]["Input"]["Size"] = 1
    e["Problem"]["Solution"]["Data"] = y_train_k
    e["Problem"]["Solution"]["Size"] = 1

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
    e["Solver"]["Data"]["Input"]["Shuffel"] = False
    e["Solver"]["Data"]["Training"]["Shuffel"] = False
    e["Solver"]["Neural Network"]["Engine"] = args.engine
    e["Solver"]["Optimizer"]["Type"] = "learner/deepSupervisor/optimizers/f"+args.optimizer

# Set Model ================================================================================
    regressor(e, len(X_train_k[0]))
# ==========================================================================================
    e["Console Output"]["Frequency"] = 1
    e["Console Output"]["Verbosity"] = "Normal"
    e["File Output"]["Enabled"] = args.save
    e["File Output"]["Frequency"] = 0
    e["File Output"]["Path"] = paths[key]
    # e["Save Only"] = ["Current Generation" ,"Run ID", "Solver"]
    e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
    k["Conduit"]["Type"] = "Sequential"
    if key == keys[1]:
        e["Solver"]["Regularizer"]["Save"] = True
        e["Solver"]["Regularizer"]["Type"] = args.regularizerType
        e["Solver"]["Regularizer"]["Coefficient"] = args.regularizerCoefficient
    eList.append(e)
k.run(eList)
# Store X_train, y_train, X_val, y_val (might change from input due to validaiton split) ===============
data = {}
for i, key in enumerate(current_keys):
    data[key] = {
        "X_train": eList[i]["Problem"]["Input"]["Data"],
        "y_train": eList[i]["Problem"]["Solution"]["Data"],
        "X_val": eList[i]["Problem"]["Data"]["Validation"]["Input"],
        "y_val": eList[i]["Problem"]["Data"]["Validation"]["Solution"]
    }
# Predict on the unseen test set =======================================================================
for i, key in enumerate(current_keys):
    # # ### Obtaining inferred results from the NN and comparing them to the actual solution
    eList[i]["Solver"]["Mode"] = "Testing"
    eList[i]["Problem"]["Testing Batch Size"] = args.testingSetSize
    eList[i]["Problem"]["Input"]["Data"] = X_test_k
    eList[i]["Problem"]["Solution"]["Data"] = y_test_k
    eList[i]["File Output"]["Enabled"] = args.save
    eList[i]["File Output"]["Path"] = os.path.join(paths[key], "predict")
k.run(eList)
# Plotting Training Results
if (args.plot):
    regularizer_type = ["","$+ \\lambda||w||^2$"]
    # Plot ture function, training data, validation data ===================================================
    fig, axes = plt.subplots(len(current_keys))
    if len(current_keys) == 1:
        axes = [axes]
    for i, key in enumerate(current_keys):
        axes[i].plot(np.linspace(x_min, x_max, nb_x_values), f(np.linspace(x_min, x_max, nb_x_values)), label="$f(x)$", color = f_c)
        axes[i].scatter(data[key]["X_train"], data[key]["y_train"], label="$y_{\\textrm{train}}=f(x)+\\epsilon$", color=train_c)
        axes[i].scatter(data[key]["X_val"], data[key]["y_val"], label="$y_{\\textrm{val}}=f(x)+\\epsilon$", color=val_c)
        axes[i].scatter(eList[i]["Problem"]["Input"]["Data"], eList[i]["Problem"]["Solution"]["Data"], label="$y_{\\textrm{test}}=f(x)+\\epsilon$", color=test_c)

    # Obtain predctions of the nn model to plot its function ===============================================
    for i, key in enumerate(current_keys):
        # # ### Obtaining inferred results from the NN and comparing them to the actual solution
        eList[i]["Solver"]["Mode"] = "Predict"
        eList[i]["Problem"]["Testing Batch Size"] = nb_x_values
        eList[i]["Problem"]["Input"]["Data"] = X_space_k
        eList[i]["File Output"]["Enabled"] = False
        eList[i]["File Output"]["Enabled"] = args.save
        eList[i]["File Output"]["Path"] = os.path.join(paths[key], "predict_func")
    k.run(eList)
    # Calculate the size of the weights ====================================================================
    yhat = []
    weight_sum = []
    for i, _ in enumerate(eList):
        yhat.append([ y[0] for y in eList[i]["Solver"]["Evaluation"] ])
        weight_sum.append(sum([ w**2 for w in eList[i]["Solver"]["Hyperparameters"] ]))

    # Plot the predicted function of the nn model ==========================================================
    for i, ax in enumerate(axes):
        if current_keys[i] == keys[0]:
            coef_size = ""
            reg_t = regularizer_type[0]
        else:
            coef_size = " and $\\lambda = $" + f'{args.regularizerCoefficient:.4f}'
            reg_t = regularizer_type[1]
        ax.plot(X_space, yhat[i], label="$\\hat{y}(x)$ "+reg_t+" with $||w||^2 = $ " + f'{weight_sum[i]:.4f}'+coef_size, color=model_c)
        ax.legend()
        ax.set_yscale("linear")
    for key in current_keys:
        SAVE_PLOT = "None"
        main(paths[key], False, SAVE_PLOT, False, ["--yscale", "linear"])
    plt.show()
print_header(width=140)

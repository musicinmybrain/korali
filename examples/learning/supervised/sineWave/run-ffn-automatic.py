import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
from korali.auxiliar.printing import *
from korali.plot.__main__ import main
import argparse
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
    default=2050,
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
    default=0.0001,
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
    default=10,
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
    help='Total size of the training set.',
    default=500,
    required=False)
parser.add_argument(
    '--trainingBatchSize',
    help='Batch size to use for training data',
    default=20,
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
    default=80,
    required=False)
parser.add_argument(
    '--testBatchSize',
    help='Batch size to use for test data',
    default=10,
    required=False)
parser.add_argument(
    '--testMSEThreshold',
    help='Threshold for the testing MSE, under which the run will report an error',
    default=0.05,
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

# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp

print_header('Korali', color=bcolors.HEADER, width=140)
print_args(vars(args), sep=' ', header_width=140)

scaling = 5.0
np.random.seed(0xC0FFEE)

# The input set has scaling and a linear element to break symmetry
trainingInputSet = np.random.uniform(0, 2 * np.pi, args.trainingSetSize)
trainingSolutionSet = np.tanh(np.exp(np.sin(trainingInputSet))) * scaling 
# Convert to korali format
trainingInputSet = [ [ [ i ] ] for i in trainingInputSet.tolist() ]
trainingSolutionSet = [ [ x ] for x in trainingSolutionSet.tolist() ]

validationInputSet = np.random.uniform(0, 2 * np.pi, args.validationBatchSize)
validationInputSet = [ [ [ i ] ] for i in validationInputSet.tolist() ]
validationSolutionSet = np.tanh(np.exp(np.sin(validationInputSet))) * scaling
validationSolutionSet = [ x[-1] for x in validationSolutionSet.tolist()]
### Defining a learning problem to infer values of sin(x)

e = korali.Experiment()

e["Problem"]["Description"] = "Supervised Learning Problem: $y(x)=\\tanh(\\exp(\\sin(\\texttt{trainingInputSet})))\\cdot\\texttt{scaling}$"
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Testing Batch Size"] = args.testBatchSize

e["Problem"]["Input"]["Data"] = trainingInputSet
e["Problem"]["Input"]["Size"] = 1
e["Problem"]["Solution"]["Data"] = trainingSolutionSet
e["Problem"]["Solution"]["Size"] = 1

e["Problem"]["Data"]["Validation"]["Input"] = validationInputSet
e["Problem"]["Data"]["Validation"]["Solution"] = validationSolutionSet
### Using a neural network solver (deep learning) for training

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Loss Function"] = "Mean Squared Error"
if args.regularizer:
    e["Solver"]["Regularizer"]["Type"] = "L2"
    e["Solver"]["Regularizer"]["Coefficient"] = args.regularizerCoefficient

e["Solver"]["Learning Rate"] = args.initialLearningRate
if args.learningRateType != "":
    e["Solver"]["Learning Rate Type"] = args.learningRateType
    e["Solver"]["Learning Rate Decay Factor"] = args.learningRateDecayFactor
    e["Solver"]["Learning Rate Lower Bound"] = args.learningRateLowerBound
    e["Solver"]["Learning Rate Steps"] = args.learningRateSteps
    e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Batch Concurrency"] = 1
e["Solver"]["Data"]["Training"]["Shuffel"] = True

### Defining the shape of the neural network
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Configuring output
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Normal"
e["File Output"]["Enabled"] = args.save
e["File Output"]["Frequency"] = 1 if args.epochs <= 100 else args.epochs/100
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
k.run(e)

# ### Obtaining inferred results from the NN and comparing them to the actual solution
testInputSet = np.random.uniform(0, 2 * np.pi, args.testBatchSize)
testInputSet = [ [ [ i ] ] for i in testInputSet.tolist() ]
testOutputSet = [ x[-1] for x in np.tanh(np.exp(np.sin(testInputSet))) * scaling ]
e["Solver"]["Mode"] = "Predict"
e["Problem"]["Input"]["Data"] = testInputSet
# ### Running Testing and getting results
e["File Output"]["Enabled"] = False
k.run(e)
# testInferredSet = [ x[0] for x in e["Solver"]["Evaluation"] ]
# print("training finished")

# # ### Calc MSE on test set
# mse = np.mean((np.array(testInferredSet) - np.array(testOutputSet))**2)
# print("MSE on test set: {}".format(mse))

# if (mse > args.testMSEThreshold):
#  print("Fail: MSE does not satisfy threshold: " + str(args.testMSEThreshold))
#  print_header(width=140)
#  exit(-1)

# ### Plotting Results
if (args.plot):
    SAVE_PLOT = False
    main("_korali_result", False, SAVE_PLOT, False, ["--yscale", "linear"])

print_header(width=140)

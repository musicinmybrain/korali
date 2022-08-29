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
from utilities import make_parser
from utilities import train_epoch, test_epoch
from sklearn.preprocessing import OneHotEncoder
from korali.plot.__main__ import main
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
    default='Automatic',
    choices=["Training", "Automatic", "Predict", "Plot"],
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
palette = sns.color_palette("deep")
train_c = palette[1]
val_c = palette[3]
lr_c = palette[5]
test_c = palette[4]
f_c = palette[0]
# plt.rcParams['text.usetex'] = True
add_dimension_to_elements = lambda l : [ [y] for y in l]
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
if args.model == "Logistic Regression":
    from linear_clf import LogisticRegression as classifier
elif args.model == "ffnn300-100":
    from linear_clf import FFNN_300_100 as classifier
elif args.model == "lenet1":
    from lenet import LeNet1 as classifier
else:
    sys.exit(f"{args.model} is not a valid model.")
#  ===========================================================================
k = korali.Engine()
e = korali.Experiment()
### Hyperparameters
MAX_RGB = 255.0
### Lading data ==============================================================================
###Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]
### Total 60 000 training samples
### Lading data ==============================================================================
t0 = time.time_ns()
mndata = MNIST("./_data")
mndata.gz = True
trainingImages, trainingLabels = mndata.load_training()
### One hot encode the labels
onehot_encoder = OneHotEncoder(sparse=False)
trainingLabels = onehot_encoder.fit_transform(add_dimension_to_elements(trainingLabels)).tolist()
trainingSet = list(zip(trainingImages, trainingLabels))
### Normalize, shuffel and split data ========================================================
random.shuffle(trainingSet)
# trainingImages, trainingLabels = trainingSet
testingImages, testingLabels = mndata.load_testing()
testingLabeles = onehot_encoder.transform(add_dimension_to_elements(testingLabels)).tolist()
# ===============================================
loading_time = (time.time_ns()-t0) / (float) (10 ** 9) # convert to floating-point second
args.dataLoadTime = f"{loading_time} s"
### Get dimensions and sizes
img_width = img_height = 28
img_size = len(trainingImages[0])
label_size = len(trainingLabels[0])
input_channels = 1
classes = 10
assert img_width*img_height == img_size
args.img_width = img_width
args.img_height = img_height
args.img_size = img_height*img_width
args.label_size = label_size
### Print Header
if args.verbosity in ("Normal", "Detailed"):
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)
# Calculate number of samples that is fitting to the BS =====================
nb_training_samples = len(trainingSet)
nb_training_samples = int((nb_training_samples*(1-args.validationSplit))/args.trainingBS)*args.trainingBS
if args.test:
    nb_training_samples = args.trainingBS*10
nb_validation_samples = int((len(trainingSet)*args.validationSplit)/args.testingBS)*args.testingBS
if args.test:
    nb_validation_samples = args.validationBS*256
if args.verbosity in ["Normal", "Detailed"]:
    print(f'{nb_training_samples} training samples')
    print(f'Discarding {int(len(trainingSet)*(1-args.validationSplit)-nb_training_samples)} training samples')
    print(f'{nb_validation_samples} validation samples')
    print(f'Discarding {int(len(trainingSet)*args.validationSplit-nb_validation_samples)} validation samples')
# nb_training_samples = 256*3
# nb_validation_samples = 256*1
trainingSet = trainingSet[:(nb_training_samples+nb_validation_samples)]
trainingSet = [([p/MAX_RGB for p in img], label) for img, label in trainingSet]
testingImages = [[p/MAX_RGB for p in img] for img in testingImages]
# Split train data into validation and train data ==============================================
trainingImages, trainingLabels = list(zip(*trainingSet[nb_validation_samples:]))
validationImages, validationLabels = zip(*trainingSet[:nb_training_samples])
nb_training_samples = len(trainingImages)
assert len(validationImages) % args.testingBS == 0
assert len(trainingImages) % args.trainingBS == 0
### Load Previous model if desired
results_dir = os.path.join("_korali_result", args.mode, args.model)
results_file = os.path.join(results_dir, "latest")
isStateFound = False
if args.load_model:
    args.validationSplit = 0.0
    isStateFound = e.loadState(results_file)
    if not isStateFound:
        sys.exit(f"No model file for {results_file} found")
    if isStateFound and args.verbosity in ["Normal", "Detailed"]:
        print("[Script] Evaluating previous run...\n")
k["Conduit"]["Type"] = "Sequential"
### Configuring general problem settings
e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Problem"]["Training Batch Size"] = args.trainingBS
# e["Problem"]["Testing Batch Sizes"] = [1, args.testingBS]
e["Problem"]["Testing Batch Size"] = args.testingBS
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = label_size
#classes
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs

e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Learning Rate Type"] = args.learningRateType
e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Loss Function"] = "Cross Entropy"
e["Solver"]["Metrics"]["Type"] = "Accuracy"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# MODEL DEFINTION ================================================================================
classifier(e, img_width, img_height, input_channels)
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
e["Save Only"] = ["Current Generation" ,"Run ID", "Results", "Solver"]

# ============================================================================
#  Automatic Training ========================================================
# ============================================================================
if args.mode in ["Automatic"]:
    ### Using a neural network solver (deep learning) for inference
    e["Problem"]["Validation Batch Size"] = args.validationBS
    e["Problem"]["Input"]["Data"] = add_dimension_to_elements(trainingImages)
    e["Problem"]["Solution"]["Data"] = trainingLabels
    e["Problem"]["Data"]["Validation"]["Input"] = add_dimension_to_elements(validationImages)
    e["Problem"]["Data"]["Validation"]["Solution"] = validationLabels
    e["Solver"]["Mode"] = "Automatic Training"
    k.run(e)

#  Training ==================================================================
elif args.mode == "Training":
    ### Converting images to Korali form (requires a time dimension)
    trainingImages = add_dimension_to_elements(trainingImages)
    validationImages = add_dimension_to_elements(validationImages)
    #  Define the training and testing loop ======================================
    #  Training Loop =============================================================
    history={'Training Loss':[],'Validation Loss':[], 'Time per Epoch': []}
    if args.mode in "Training":
        for epoch in range(args.epochs):
            tp_start = time.time()
            saveLastModel = True if (epoch == args.epochs-1 and args.save) else False
            # saveLastModel = False
            train_loss = train_epoch(e, k, args, (trainingImages, trainingLabels), saveLastModel)
            tp = time.time()-tp_start
            val_loss = test_epoch(e, k, args, (validationImages, validationLabels))
            if args.verbosity in ["Normal", "Detailed"]:
                print(f'EPOCH {epoch + 1}/{args.epochs} {tp:.3f}s \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}')
            history['Training Loss'].append(train_loss)
            history['Time per Epoch'].append(tp)
    if args.save:
        results_file = os.path.join(results_dir, "history.json")
        with open(results_file, 'w') as file:
            file.write(json.dumps(history))
# =============================================================================================

# PREDICTING ==================================================================================
e["File Output"]["Enabled"] = False
if args.mode in ["predict"]:
    if not isStateFound:
        sys.exit("Cannot predict without loading or training a model.")
    testImage = trainingImages[:args.testingBS]
    e["Problem"]["Input"]["Data"] = add_dimension_to_elements(testImage)
    e["Problem"]["Solution"]["Data"] = testImage
    e["Solver"]["Mode"] = "Testing"
    k.run(e)

# # Plotting Results
if args.plot:
    #  Plot Original Ground Truth ==================================================
    arr_to_img = lambda img : np.reshape(img, (img_height, img_width))
    fig, axes = plt.subplots(nrows=4, ncols=3)
    for row in axes:
        for ax in row:
            X, y  = random.choice(list(zip(trainingImages, trainingLabels)))
            number = onehot_encoder.inverse_transform([y]).item()
            ax.set_title(number)
            ax.imshow(arr_to_img(X), cmap='gist_gray')
            ax.set_xlabel(np.array(y).astype(int))
    # fig.su# set the spacing between subplots
    plt.subplots_adjust(bottom=0.1,
                        top=0.95,
                        hspace=0.4)
    # Plot Losses ======================================================================
    # Note: need to run run-classification.py with --save flag once to get output files.
    SAVE_PLOT = "None"
    main(results_dir, False, SAVE_PLOT, False, ["--yscale", "linear"])
    plt.show()

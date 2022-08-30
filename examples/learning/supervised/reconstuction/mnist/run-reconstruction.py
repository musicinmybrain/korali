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
    choices=["Training", "Automatic", "Predict"],
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

add_time_dimension = lambda l : [ [y] for y in l]
args = parser.parse_args()
k = korali.Engine()
e = korali.Experiment()
### Hyperparameters
MAX_RGB = 255.0
### Lading data ==============================================================================
###Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]
### Total 60 000 training samples
### Lading data ==============================================================================
t0 = time.time_ns()
mndata = MNIST("../../data/mnist")
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()
loading_time = (time.time_ns()-t0) / (float) (10 ** 9) # convert to floating-point second
args.dataLoadTime = f"{loading_time} s"
### Get dimensions and sizes
img_width = img_height = 28
img_size = len(trainingImages[0])
input_channels = 1
assert img_width*img_height == img_size
args.img_width = img_width
args.img_height = img_height
args.img_size = img_height*img_width
### Normalize, shuffel and split data ========================================================
random.shuffle(trainingImages)
# Calculate number of samples that is fitting to the BS =====================
nb_training_samples = len(trainingImages)
nb_training_samples = int((nb_training_samples*(1-args.validationSplit))/args.trainingBS)*args.trainingBS
if args.test:
    nb_training_samples = args.trainingBS*10
nb_validation_samples = int((len(trainingImages)*args.validationSplit)/args.testingBS)*args.testingBS
if args.test:
    nb_validation_samples = args.validationBS*1
if args.verbosity in ["Normal", "Detailed"]:
    print(f'{nb_training_samples} training samples')
    print(f'Discarding {int(len(trainingImages)*(1-args.validationSplit)-nb_training_samples)} training samples')
    print(f'{nb_validation_samples} validation samples')
    print(f'Discarding {int(len(trainingImages)*args.validationSplit-nb_validation_samples)} validation samples')
trainingImages = trainingImages[:(nb_training_samples+nb_validation_samples)]
trainingImages = [[p/MAX_RGB for p in img] for img in trainingImages]
testingImages = [[p/MAX_RGB for p in img] for img in testingImages]
# Split train data into validation and train data ==============================================
validationImages = trainingImages[:nb_validation_samples]
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
if args.verbosity in ["Normal", "Detailed"]:
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)
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
e["Problem"]["Testing Batch Sizes"] = [1, args.testingBS]
e["Problem"]["Testing Batch Size"] = args.testingBS
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs

e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Learning Rate Type"] = args.learningRateType
e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# MODEL DEFINTION ================================================================================
if args.model == "linear":
    from linear_autoencoder import configure_autencoder as autoencoder
elif args.model == "simple":
    from cnn_autoencoder import simple_cnn_autoencoder as autoencoder
else:
    sys.exit(f"No valid model '{args.model}' selected")
autoencoder(e, img_width, img_height, args.latentDim, input_channels)
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

#  Automatic Training ========================================================
if args.mode in ["Automatic"]:
    ### Using a neural network solver (deep learning) for inference
    e["Problem"]["Validation Batch Size"] = args.validationBS
    # e["Solver"]["Data"]["Validation"]["Split"] = args.validationSplit
    e["Problem"]["Input"]["Data"] = add_time_dimension(trainingImages)
    e["Problem"]["Solution"]["Data"] = trainingImages
    e["Problem"]["Data"]["Validation"]["Input"] = add_time_dimension(validationImages)
    e["Problem"]["Data"]["Validation"]["Solution"] = validationImages
    e["Solver"]["Mode"] = "Automatic Training"
    k.run(e)
#  Training ==================================================================
elif args.mode == "Training":
    ### Converting images to Korali form (requires a time dimension)
    trainingImages = add_time_dimension(trainingImages)
    validationImages = add_time_dimension(validationImages)
    def loss_MSE(yhat_batch, y_batch):
        squaredMeanError = 0.0
        for yhat, y in list(zip(yhat_batch, y_batch)):
            for yhat_val, y_val in list(zip(yhat, y)):
                diff = yhat_val - y_val
                squaredMeanError += diff * diff
            squaredMeanError = squaredMeanError / (float(len(yhat)))
        return squaredMeanError

    #  Define the training and testing loop ======================================
    def train_epoch(e, save_last_model = False):
        e["Solver"]["Mode"] = "Training"
        # Set train mode for both the encoder and the decoder
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        train_loss = []
        stepsPerEpoch = int(len(trainingImages) / args.trainingBS)
        for step in range(stepsPerEpoch):
            # Creating minibatch
            image_batch = trainingImages[step * args.trainingBS : (step+1) * args.trainingBS] # N x T x C
            y = [ x[0] for x in image_batch ] # N x C
            # Passing minibatch to Korali
            if save_last_model and step == stepsPerEpoch-1:
                e["File Output"]["Enabled"] = True
            else:
                e["File Output"]["Enabled"] = False
            e["Problem"]["Input"]["Data"] = image_batch
            e["Problem"]["Solution"]["Data"] = y
            k.run(e)
            # yhat = e["Solver"]["Evaluation"]
            # loss = loss_MSE(y, yhat)
            # print(f'Step {step+1} / {stepsPerEpoch}\t Training Loss {loss}')
            loss = e["Results"]["Training Loss"]
            train_loss.append(loss)
        return np.mean(train_loss).item()

    def test_epoch(e):
        e["File Output"]["Enabled"] = False
        e["Problem"]["Testing Batch Size"] = args.testingBS
        e["Solver"]["Mode"] = "Testing"
        # Set train mode for both the encoder and the decoder
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        val_loss = []
        stepsPerEpoch = int(len(validationImages) / args.testingBS)
        for step in range(stepsPerEpoch):
            # Creating minibatch
            image_batch = validationImages[step * args.testingBS : (step+1) * args.testingBS] # N x T x C
            y = [ x[0] for x in image_batch ] # N x C
            # Passing minibatch to Korali
            e["Problem"]["Input"]["Data"] = image_batch
            e["Problem"]["Solution"]["Data"] = y
            # Evaluate loss
            k.run(e)
            # print(f'Step {step+1} / {stepsPerEpoch}\t Testing Loss {loss}')
            loss = e["Results"]["Testing Loss"]
            val_loss.append(loss)
        return np.mean(val_loss).item()

    #  Training Loop =============================================================
    history={'Training Loss':[],'Validation Loss':[], 'Time per Epoch': []}
    if args.mode in "Training":
        for epoch in range(args.epochs):
            tp_start = time.time()
            saveLastModel = True if (epoch == args.epochs-1 and args.save) else False
            # saveLastModel = False
            train_loss = train_epoch(e, saveLastModel)
            tp = time.time()-tp_start
            val_loss = test_epoch(e)
            # print(f'EPOCH {epoch + 1}/{args.epochs} {tp:.3f}s \t val loss {val_loss:.3f}')
            if args.verbosity in ["Normal", "Detailed"]:
                print(f'EPOCH {epoch + 1}/{args.epochs} {tp:.3f}s \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}')
            history['Training Loss'].append(train_loss)
            history['Validation Loss'].append(val_loss)
            history['Time per Epoch'].append(tp)
    if args.save:
        results_file = os.path.join(results_dir, "history.json")
        with open(results_file, 'w') as file:
            file.write(json.dumps(history))

# # PREDICTING ================================================================================
e["File Output"]["Enabled"] = False
if args.mode in ["predict"]:
    if not isStateFound:
        sys.exit("Cannot predict without loading or training a model.")
    testImage = trainingImages[:args.testingBS]
    e["Problem"]["Input"]["Data"] = add_time_dimension(testImage)
    e["Problem"]["Solution"]["Data"] = testImage
    e["Solver"]["Mode"] = "Testing"
    k.run(e)

# # Plotting Results
if args.plot:
    #  Plot Reconstruced Images ==================================================
    SAMPLES_TO_DISPLAY = 8
    arr_to_img = lambda img : np.reshape(img, (img_height, img_width))
    fig, axes = plt.subplots(nrows=SAMPLES_TO_DISPLAY, ncols=2)
    random.shuffle(testingImages)
    e["Problem"]["Testing Batch Size"] = args.testingBS
    e["Solver"]["Mode"] = "Predict"
    y = [random.choice(testingImages) for i in range(args.testingBS)]
    e["Problem"]["Input"]["Data"] = add_time_dimension(y)
    k.run(e)
    yhat = e["Solver"]["Evaluation"]
    for y, yhat, ax in list(zip(y[:SAMPLES_TO_DISPLAY], yhat[:SAMPLES_TO_DISPLAY], axes)):
        ax[0].imshow(arr_to_img(y), cmap='gist_gray')
        ax[1].imshow(arr_to_img(yhat), cmap='gist_gray')
    # fig.tight_layout()
    if args.save:
        plt.savefig(os.path.join(results_dir, "reconstructions.png"))
    #  Plot Losses ===============================================================
    sns.set()
    with open(results_file, 'r') as f:
        results = json.load(f)
    if "Results" in results:
        results = results["Results"]
    fig, ax = plt.subplots(figsize=(8, 8))
    epochs = range(1, results["Epoch"]+1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    # ax.set_yscale(args.yscale)
    ax.semilogy(epochs, results["Training Loss"] ,'-', label="Training Loss", color=train_c)
    if 'Validation Loss' in results:
        ax.semilogy(epochs, results["Validation Loss"] ,'-', label="Validation Loss", color=val_c)
    ax.set_xlabel('Epochs')
    ylabel = results['Loss Function']
    if "Regularizer" in results:
        ylabel+= " + " + results["Regularizer"]["Type"]
    ax.set_ylabel(ylabel)
    if "Description" in results:
        ax.set_title(results["Description"].capitalize())
    plt.legend()
    if args.save:
        plt.savefig(os.path.join(results_dir, "losses.png"))
    # if 'Learning Rate' in results:
    #     ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    #     ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
    #     ax2.plot(epochs, results["Learning Rate"], color=lr_c)
    #     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    with open(results_file, 'r') as f:
        results = json.load(f)
    if "Results" in results:
        results = results["Results"]
    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    epochs = range(1, results["Epoch"]+1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    def plot_loss(results):
        # ax.set_yscale(args.yscale)
        ax.semilogy(epochs, results["Training Loss"] ,results["style"], label="Training Loss", color=train_c)
        if 'Validation Loss' in results:
            ax.semilogy(epochs, results["Validation Loss"] ,results["style"], label="Validation Loss", color=val_c)
            ax.set_xlabel('Epochs')
        ylabel = results['Loss Function']
        if "Regularizer" in results:
            ylabel+= " + " + results["Regularizer"]["Type"]
            ax.set_ylabel(ylabel)
        if "Description" in results:
            ax.set_title(results["Description"].capitalize())
        plt.legend()
        # if 'Learning Rate' in results:
        #     ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        #     ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
        #     ax2.plot(epochs, results["Learning Rate"], color=lr_c)
        #     fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if args.verbosity in ["Normal", "Detailed"]:
        print_header(width=140)

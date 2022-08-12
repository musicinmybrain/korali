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
from korali.plot.__main__ import main
from mnist import MNIST
sys.path.append(os.path.abspath('./_models'))
sys.path.append(os.path.abspath('..'))
from autoencoder import configure_autencoder
from utilities import make_parser
#  Arguments =================================================================
parser = make_parser()
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
# ==================================================================================
# In case of
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp
print_header('Korali', color=bcolors.HEADER, width=140)
add_time_dimension = lambda l : [ [y] for y in l]
k = korali.Engine()
e = korali.Experiment()
### Hyperparameters
MAX_RGB = 255.0
### Lading data ==============================================================================
t0 = time.time_ns()
mndata = MNIST("./_data")
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()
### ==========================================================================================
img_width = 28
img_height = 28
img_size = len(trainingImages[0])
input_channels = 1
assert img_width*img_height == img_size
args.img_width = img_width
args.img_height = img_height
args.img_size = img_height*img_width
# Normalize Training Images
random.shuffle(trainingImages)
#  Calculate number of samples that is fitting to the BS =====================
nb_training_samples = len(trainingImages)
nb_training_samples = 256*5
nb_training_samples = int((nb_training_samples-nb_training_samples*args.validationSplit)/args.trainingBS)*args.trainingBS
nb_validation_samples = int(int(nb_training_samples*args.validationSplit)/args.testingBS)*args.testingBS
trainingImages = trainingImages[:(nb_training_samples+nb_validation_samples)]
trainingImages = [[p/MAX_RGB for p in img] for img in trainingImages]
### Converting images to Korali form (requires a time dimension)
trainingImages = add_time_dimension(trainingImages)
testingImages = add_time_dimension(testingImages)
# Split train data into validation and train data ==============================================
validationImages = trainingImages[:nb_training_samples]
trainingImages = trainingImages[nb_validation_samples:]
nb_training_samples = len(trainingImages)
assert len(validationImages) % args.testingBS == 0
assert len(trainingImages) % args.trainingBS == 0
### Print Args
print_args(vars(args), sep=' ', header_width=140)
### Load Previous model if desired
isStateFound = False
if args.load_model:
    args.validationSplit = 0.0
    isStateFound = e.loadState("_korali_result/latest")
    if not isStateFound:
        sys.exit("No model file for _korali_result/latest found")
    if(isStateFound):
        print("[Script] Evaluating previous run...\n")

k["Conduit"]["Type"] = "Sequential"
### Configuring general problem settings
e["Problem"]["Type"] = "Supervised Learning"
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Problem"]["Training Batch Size"] = args.trainingBS
e["Problem"]["Testing Batch Size"] = args.testingBS
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size
### Using a neural network solver (deep learning) for inference
# e["Solver"]["Termination Criteria"]["Epochs"] = 30
e["Solver"]["Termination Criteria"]["Max Generations"] = 1

e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Learning Rate Type"] = args.learningRateType
e["Solver"]["Learning Rate Save"] = True

e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# MODEL DEFINTION ================================================================================
configure_autencoder(e, img_width, img_height, input_channels, args.latentDim)
# ================================================================================
### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
e["Random Seed"] = args.seed
e["File Output"]["Enabled"] = False
e["File Output"]["Path"] = "_korali_result"
# e["Save"]["Problem"] = False
# e["Save"]["Solver"] = False

#  Define the training and testing loop ======================================
def train_epoch(e):
    e["Solver"]["Mode"] = "Training"
    # Set train mode for both the encoder and the decoder
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    stepsPerEpoch = int(len(trainingImages) / args.trainingBS)
    for step in range(stepsPerEpoch):
        # Creating minibatch
        image_batch = trainingImages[step * args.trainingBS : (step+1) * args.trainingBS] # N x T x C
        imgage_batch_target = [ x[0] for x in image_batch ] # N x C
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = image_batch
        e["Problem"]["Solution"]["Data"] = imgage_batch_target
        k.run(e)
        e["Solver"]["Termination Criteria"]["Max Generations"] = e["Solver"]["Termination Criteria"]["Max Generations"]+1
        loss = e["Results"]["Training Loss"]
        print(f'Step {step+1} / {stepsPerEpoch}\t Loss {loss}')
        train_loss.append(loss)
    return np.mean(train_loss).item()

def test_epoch(e):
    e["Solver"]["Mode"] = "Testing"
    # Set train mode for both the encoder and the decoder
    validation_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    # import pdb; pdb.set_trace()
    stepsPerEpoch = int(len(validationImages) / args.testingBS)
    for step in range(stepsPerEpoch):
        # Creating minibatch
        image_batch = validationImages[step * args.testingBS : (step+1) * args.testingBS] # N x T x C
        imgage_batch_target = [ x[0] for x in image_batch ] # N x C
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = image_batch
        e["Problem"]["Solution"]["Data"] = imgage_batch_target
        # Evaluate loss
        # loss = loss_fn(decoded_data, image_batch)
        e["Solver"]["Termination Criteria"]["Max Generations"] = e["Solver"]["Termination Criteria"]["Max Generations"]+1
        k.run(e)
        print(f'Testing Loss {e["Results"]["Testing Loss"]}')
        validation_loss.append(e["Results"]["Testing Loss"])
    return np.mean(validation_loss).item()

history={'train_loss':[],'val_loss':[], 'time_per_epoch': []}
if args.mode in ["all", "train"]:
    for epoch in range(args.epochs):
        tp_start = time.time()
        train_loss = train_epoch(e)
        tp = time.time()-tp_start
        val_loss = test_epoch(e)
        print(f'EPOCH {epoch + 1}/{args.epochs} {tp:.3f}s \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['time_per_epoch'].append(tp)

if args.save:
    with open('_results/latest', 'w') as file:
        file.write(json.dumps(history))
if args.plot:
    plt.figure(figsize=(10,8))
    plt.semilogy(history['train_loss'], label='Train')
    plt.semilogy(history['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()

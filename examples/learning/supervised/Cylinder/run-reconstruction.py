#!/usr/bin/env python
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import korali
import shutil
import time
from mpi4py import MPI
sys.path.append('./_models')
sys.path.append('./_scripts')
from cnn_autoencoder import configure_cnn_autencoder
from autoencoder import configure_autencoder
from utilities import min_max_scalar
from korali.auxiliar.printing import *
from korali.plot.__main__ import main
from utilities import get_last_timestep
from utilities import get_minibatch
from utilities import move_dir
from utilities import copy_dir
from utilities import make_parser
from utilities import initialize_constants
from utilities import exp_dir_str
from utilities import DataLoader
import utilities as constants

initialize_constants()
CWD = os.getcwd()
REL_ROOT = os.path.relpath("/")
TIMESTEPS = 0
parser = make_parser()
isMaster = lambda: args.conduit != constants.DISTRIBUTED or (args.conduit == constants.DISTRIBUTED and MPIrank == MPIroot)

iPython = True
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        ipython = True

args = parser.parse_args()

if args.verbosity != constants.SILENT and isMaster():
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)

# = Initalize Korali Engine and Experiment ==============================
k = korali.Engine()
# Sets ranks in case of distributed computing ============================
if args.conduit == constants.DISTRIBUTED:
    MPIcomm = MPI.COMM_WORLD
    MPIrank = MPIcomm.Get_rank()
    MPIsize = MPIcomm.Get_size()
    MPIroot = MPIsize - 1
    k.setMPIComm(MPI.COMM_WORLD)
# ===================== Loading the data =================================
loader = DataLoader(args)
loader.fetch_data()
X_train = loader.X_train
X_test = loader.X_test
nb_train_samples = len(X_train)
nb_test_samples = len(X_test)
timesteps, input_channels, img_width, img_height = loader.shape
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
# = permute
# ===================== Preprocessing ====================================
# TODO make this generic and adapt to general time series with preprocessing
# ONLY FOR one time point at the moment
X_train = np.squeeze(X_train, axis=1)
X_test = np.squeeze(X_test, axis=1)
# TODO finished
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
# TODO make this generic and adapt to general time series with preprocessing
X_train = np.expand_dims(X_train, axis=1).tolist()
X_test = np.expand_dims(X_test, axis=1).tolist()
# TODO finished
# ================ Getting the ground truth ==============================
y_train = get_last_timestep(X_train)
y_test = get_last_timestep(X_test)
assert np.shape(y_train[0]) == np.shape(y_test[0])
input_size = output_size = len(y_train[0])

if args.test:
    # args.epochs = 30
    # nb_training_samples = 260
    # args.trainingBatchSize = 50
    # trainingImages = trainingImages[:nb_training_samples]
    # args.validationSplit = 60
    pass

# Create Result dirs =========================================================
# experiment dir from _korali_result
EXPERIMENT_DIR = exp_dir_str(args)
RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
RESULT_DIR_WITHOUT_SCRATCH = os.path.relpath(RESULT_DIR, constants.SCRATCH)
RESULT_DIR_ON_HOME = os.path.join(constants.HOME, RESULT_DIR_WITHOUT_SCRATCH)

e = korali.Experiment()

if args.file_output:
    # Note: korali appends ./ => requires relative path i.e. ../../../..
    e["File Output"]["Path"] = RESULT_DIR
    if isMaster():
        os.makedirs(RESULT_DIR, exist_ok=True)
        if constants.SCRATCH:
            os.makedirs(RESULT_DIR_ON_HOME, exist_ok=True)

# Loading previous models ==================================================
# isStateFound = e.loadState(os.path.join(RESULT_DIR, "/latest"))
# if isMaster() and isStateFound and args.verbosity != constants.SILENT:
#     print("[Script] Evaluating previous run...\n")
e["Random Seed"] = 0xC0FFEE
k["Conduit"]["Type"] = args.conduit

e["Problem"]["Description"] = "Autoencoder - Flow behind the cylinder."
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = TIMESTEPS+1
e["Problem"]["Training Batch Size"] = args.training_batch_size
# e["Problem"]["Testing Batch Size"] = testingBatchSize

e["Problem"]["Input"]["Data"] = X_train
e["Problem"]["Input"]["Size"] = timesteps*input_channels*img_width*img_height
e["Problem"]["Solution"]["Data"] = y_train
e["Problem"]["Solution"]["Size"] = input_channels*img_width*img_height

# Solver =====================================================
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Batch Concurrency"] = args.batch_concurrency
e["Solver"]["Data"]["Validation"]["Split"] = args.validation_split
# Learning Rate =====================================================
e["Solver"]["Learning Rate"] = args.initialLearningRate
if args.learningRateType != "":
    e["Solver"]["Learning Rate Type"] = args.learningRateType
    e["Solver"]["Learning Rate Decay Factor"] = args.learningRateDecayFactor
    e["Solver"]["Learning Rate Lower Bound"] = args.learningRateLowerBound
    e["Solver"]["Learning Rate Steps"] = args.learningRateSteps
    e["Solver"]["Learning Rate Save"] = True
# ====================================================================
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
# e["Problem"]["Input"]["Data"] = X_test
# e["Problem"]["Solution"]["Data"] = y_test

# ===================== Model Selection ====================================
# if args.model == constants.AUTOENCODER:
#     configure_autencoder(e, img_width, img_height, TIMESTEPS, input_channels, args.latent_dim)
# else:
#     configure_cnn_autencoder(e, args.latent_dim, img_width, img_height, input_channels)

input_size = output_size = img_width*img_height*input_channels
img_height_red = img_height/2
img_width_red = img_width/2
# ===================== Input Layer
e["Problem"]["Input"]["Size"] = input_size
# ===================== Down Sampling
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Resampling Type"] = "Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Resampling"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"] = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"] = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Width"] = img_width_red
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Height"] = img_height_red
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = input_channels*img_height_red*img_width_red
# ===================== Encoder
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = args.latent_dim
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Elementwise/ReLU"
##  =================== Decoder
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = input_channels*img_height_red*img_width_red
# ===================== Up Sampling
e["Solver"]["Neural Network"]["Output Layer"]["Resampling Type"] = "Linear"
e["Solver"]["Neural Network"]["Output Layer"]["Type"] = "Layer/Resampling"
e["Solver"]["Neural Network"]["Output Layer"]["Image Width"] = img_width_red
e["Solver"]["Neural Network"]["Output Layer"]["Image Height"] = img_height_red
e["Solver"]["Neural Network"]["Output Layer"]["Output Width"] = img_width
e["Solver"]["Neural Network"]["Output Layer"]["Output Height"] = img_height
e["Solver"]["Neural Network"]["Output Layer"]["Output Channels"] = input_channels*img_width*img_height
# Activation ========================
e["Solver"]["Neural Network"]["Output Activation"] = "Elementwise/Logistic"
e["Problem"]["Solution"]["Size"] = output_size

### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
### Configuring output
e["File Output"]["Enabled"] = args.file_output
e["Console Output"]["Verbosity"] = args.verbosity
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Normal"
e["File Output"]["Frequency"] = 1 if args.epochs <= 100 else args.epochs/100
e["Save"]["Problem"] = False
e["Save"]["Solver"] = False

if args.conduit == constants.DISTRIBUTED:
    k.setMPIComm(MPI.COMM_WORLD)

# TRAINING ===============================================================================
k.run(e)
# # TESTING ================================================================================
# e["Solver"]["Mode"] = "Testing"
# k.run(e)

# Move Result dir back to HOME ===========================================================
if isMaster():
    if args.file_output:
        # Writing testing error to output
        if constants.SCRATCH:
            # move_dir(RESULT_DIR, RESULT_DIR_ON_HOME)
            # copy_dir(RESULT_DIR, RESULT_DIR_ON_HOME)
            pass
# Plotting      ===========================================================================
if args.plot:
    pass
print_header(width=140)

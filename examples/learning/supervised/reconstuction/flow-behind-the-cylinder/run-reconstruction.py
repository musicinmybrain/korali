#!/usr/bin/env python
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import korali
import shutil
import time
# from mpi4py import MPI
import matplotlib.pyplot as plt
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

iPython = False
print(sys.argv[0])
if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
    # a = ['--data-type', 'test128', '--plot', '--epochs', '200', '--save', '--overwrite', '--learningRateType', 'Time Based', '--initialLearningRate', '0.01', '--learningRateDecayFactor', '0.001', '--learningRateLowerBound', '0.001']
    tmp_args = sys.argv
    sys.argv = ['']
    iPython = True
    # args = parser.parse_args(a)
    args = parser.parse_args()
else:
    args = parser.parse_args()

if iPython:
    sys.argv = tmp_args

if args.learningRateType == "" or args.learningRateType == "Const":
    del args.learningRateDecayFactor
    del args.learningRateLowerBound
    del args.learningRateSteps

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

if args.save:
    # Note: korali appends ./ => requires relative path i.e. ../../../..
    e["File Output"]["Path"] = EXPERIMENT_DIR
    if isMaster():
        if args.overwrite:
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
        os.makedirs(RESULT_DIR, exist_ok=True)
        if constants.SCRATCH:
            os.makedirs(RESULT_DIR_ON_HOME, exist_ok=True)

# Loading previous models ==================================================
isStateFound = False
if args.load_model:
    model_file = os.path.join(RESULT_DIR, 'latest')
    isStateFound = e.loadState(model_file)
    if not isStateFound:
        sys.exit(f"No model file for {model_file} found")
    if isMaster() and isStateFound and args.verbosity != constants.SILENT:
        print("[Script] Evaluating previous run...\n")
e["Random Seed"] = 0xC0FFEE
k["Conduit"]["Type"] = args.conduit

e["Problem"]["Description"] = f"{args.model} with encoder dim {args.latent_dim} trained on data {args.data_type}"
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = TIMESTEPS+1
e["Problem"]["Training Batch Size"] = args.training_batch_size
e["Problem"]["Testing Batch Size"] = nb_test_samples

e["Problem"]["Input"]["Data"] = X_train
e["Problem"]["Input"]["Size"] = timesteps*input_channels*img_width*img_height
e["Problem"]["Solution"]["Data"] = y_train
e["Problem"]["Solution"]["Size"] = input_channels*img_width*img_height

# Solver =====================================================
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Batch Concurrency"] = args.batch_concurrency
if isStateFound:
    # Use validation and input set from previous run
    e["Solver"]["Data"]["Validation"]["Split"] = 0.0
else:
    e["Solver"]["Data"]["Validation"]["Split"] = args.validation_split
# Learning Rate =====================================================
e["Solver"]["Learning Rate"] = args.initialLearningRate
if not args.learningRateType or args.learningRateType != "Const":
    e["Solver"]["Learning Rate Type"] = args.learningRateType
    e["Solver"]["Learning Rate Decay Factor"] = args.learningRateDecayFactor
    e["Solver"]["Learning Rate Lower Bound"] = args.learningRateLowerBound
    e["Solver"]["Learning Rate Steps"] = args.learningRateSteps
    e["Solver"]["Learning Rate Save"] = True
# ====================================================================
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs

# ===================== Model Selection ====================================
if args.model == constants.AUTOENCODER:
    configure_autencoder(e, img_width, img_height, TIMESTEPS, input_channels, args.latent_dim)
else:
    configure_cnn_autencoder(e, args.latent_dim, img_width, img_height, input_channels)
### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
### Configuring output
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = args.verbosity
e["File Output"]["Enabled"] = args.save
e["File Output"]["Frequency"] = 1 if args.epochs <= 100 else args.epochs/10
# Run ID required by korali plotting script.
e["Save Only"] = ["Run ID", "Current Generation", "Results", "Solver"]

if args.conduit == constants.DISTRIBUTED:
    k.setMPIComm(MPI.COMM_WORLD)

# TRAINING ===============================================================================
if args.mode in ["all", "train"]:
    e["Solver"]["Mode"] = "Automatic Training"
    k.run(e)
# PREDICTING ================================================================================
e["File Output"]["Enabled"] = False
if args.mode in ["all", "test"]:
    if args.mode == "test" and not isStateFound:
        sys.exit("Cannot predict without loading or training a model.")

    e["Problem"]["Input"]["Data"] = X_test
    e["Solver"]["Mode"] = "Predict"
    k.run(e)

# Move Result dir back to HOME ===========================================================
if isMaster():
    if args.save:
        # Writing testing error to output
        if constants.SCRATCH:
            # move_dir(RESULT_DIR, RESULT_DIR_ON_HOME)
            # copy_dir(RESULT_DIR, RESULT_DIR_ON_HOME)
            pass
# Plotting      ===========================================================================
SAMPLES_TO_DISPLAY = 4
if args.plot:
    if not args.mode in ["all", "test"]:
        sys.exit("Need to evaluate model first in order to plot.")
    arr_to_img = lambda img : np.reshape(img, (img_height, img_width))
    fig, axes = plt.subplots(nrows=SAMPLES_TO_DISPLAY, ncols=2)
    for idx, row in enumerate(axes):
        row[0].imshow(arr_to_img(e["Problem"]["Solution"]["Data"][idx]))
        row[1].imshow(arr_to_img(e["Solver"]["Evaluation"][idx]))
    SAVE_PLOT = "None"
    main(RESULT_DIR, False, SAVE_PLOT, False, ["--yscale", "linear"])
    plt.show()
print_header(width=140)


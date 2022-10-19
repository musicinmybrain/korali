#!/usr/bin/env python
import os
import sys
import numpy as np
import korali
from korali.auxiliar.printing import *
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


sys.path.append(os.path.abspath('./_models'))
sns.set()

def flops_per_layer(bs, ic, oc):
    forward_gemm = 2*ic*oc*bs
    backward_gemm = 2*forward_gemm
    forward_acitvation = bs*dim
    backward_acitvation = bs*dim
    return forward_gemm+backward_gemm+forward_acitvation+backward_acitvation

def get_flops(training_set_size, bs, dim, hidden_layers):
    layers = hidden_layers+1
    iterrations = training_set_size/bs
    flops = iterrations*layers*flops_per_layer(bs, dim, dim)
    return flops

def mkdir_p(dir):
    """Make a directory if it doesn't exist and create intermediates as well."""
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--epochs',
    help='Maximum Number of epochs to run',
    default=1,
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
    default=2**15,
    type=int,
    required=False)
parser.add_argument(
    '--trainingBS',
    help='Batch size to use for training data',
    type=int,
    default=16,
    # choices=[16, 1024, 8192, 16384, 32768],
    required=False)
parser.add_argument(
    "-s",
    "--save",
    help="Indicates if to save models to _korali_results.",
    required=False,
    action="store_true"
)
parser.add_argument(
    "--plot",
    help="Indicates if to plot the model.",
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
    default="single",
    required=False,
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
    required=False,
    choices=[15,17,19,21,23,25],
    default=21,
    type=int,
)
parser.add_argument(
    '--verbosity',
    help='Verbosity to print',
    default="Normal",
    required=False
)
parser.add_argument(
    '--threads',
    help='Verbosity to print',
    default="notset",
    required=False
)
parser.add_argument(
    '--path',
    help='Verbosity to print',
    default="",
    required=False
)
parser.add_argument(
    "--submit",
    help="Whether to run the experiment.",
    required=False,
    action="store_true"
)


threads = os.environ["OMP_NUM_THREADS"]
tf_threads = tf.config.threading.get_inter_op_parallelism_threads()
np.random.seed(0xC0FFEE)
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython", "/usr/lib/python3.10/site-packages/ipykernel_launcher.py"]:
        #sys.argv = ['--model', 'multi', "--weight", "21"]
        sys.argv = []
        IPYTHON = True

args = parser.parse_args()
#sys.argv = tmp

# Saving Path ================================================================
if args.verbosity in ["Normal", "Detailed"]:
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)

# Models Size Definitions ====================================================
SMALLEST_LAYER_SIZE_EXPONENT = 7
SMALLEST_LAYER_SIZE = 2**SMALLEST_LAYER_SIZE_EXPONENT
OUTPUT_LAYERS = 1
dim=SMALLEST_LAYER_SIZE
nb_of_layers = 2**(args.weight_dim-(2*SMALLEST_LAYER_SIZE_EXPONENT+OUTPUT_LAYERS))
if args.verbosity in ("Normal", "Detailed"):
    print(f"layer dimension 2^{SMALLEST_LAYER_SIZE_EXPONENT}")
    print(f"layers {nb_of_layers}")
    print(f"Tensorflow number of threads {tf_threads} (0 = let system decide)")

# Model Definition ===========================================================
class LinBench(tf.keras.Model):

    def __init__(self, nb_of_layers, layer_size):
        super().__init__()
        self.layer_pipeline = []
        #self.layer_pipeline.append(layers.Input(shape=(layer_size,), name="Input_Layer")) # add the first layer
        self.layer_pipeline.append(layers.Dense(layer_size, activation="relu", input_shape=[layer_size]))
        for idx in range(nb_of_layers-1):
            self.layer_pipeline.append(layers.Dense(layer_size, activation="relu", name=f"Layer_{idx+1}"))

    @tf.function
    def call(self, input, training=False):
        ## Input layer
        x = input
        for layer in self.layer_pipeline:
            x = layer(x)
        return x

test = LinBench(dim, SMALLEST_LAYER_SIZE)
# Sequential API as Alternative ==============================================
# layer_pipeline = []
# layer_size = dim
# layer_pipeline.append(layers.Input(shape=(layer_size,))) # add the first layer
# for l in range(nb_of_layers-1):
#     layer_pipeline.append(layers.Dense(layer_size, activation="relu"))
# test = keras.Sequential(layer_pipeline)


# Loss function and Optimizer ================================================
optimizer = tf.keras.optimizers.Adam(args.initialLearningRate)
test.compile(optimizer=optimizer, loss="mse")

# Create Datasets ============================================================
np.random.seed(0xC0FFEE)
X_train = np.random.rand(args.trainingSetSize, dim)
y_train = np.random.rand(args.trainingSetSize, dim)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(args.trainingBS)

# Define Training Loop =======================================================
args.submit = True
args.epoch = 1
if args.submit:
    duration = time.time_ns()
    tf_verbosity = 1 if args.verbosity in ["Normal", "Detailed"] else 0
    history_training = test.fit(train_dataset, epochs=args.epochs, verbose=tf_verbosity)
    duration = time.time_ns()-duration
    runtime_per_epochs = duration/args.epochs*10**-9
    print(f"Average Runtime of {runtime_per_epochs}s per Epoch")
    # return the history of training process
    pd.DataFrame(history_training.history)
    if args.save:
        mkdir_p(path)
        with open(os.path.join(path, "results.csv"), 'w') as fb:
            fb.write(str(runtime_per_epochs))

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
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import _pin_memory, nn
import torch.optim as optim
import seaborn as sns
import torch.jit as jit

sys.path.append(os.path.abspath('./_models'))
sns.set()


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
    default=32,
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
threads = os.environ["OMP_NUM_THREADS"]
torch.set_num_threads(72)
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

if args.verbosity in ["Normal", "Detailed"]:
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)

# Models
# ============================================================================

if not args.path:
    path = os.path.join("_korali_result", args.other, "pytorch", f"model_{args.model}", f"BS_{args.trainingBS}", f"WeightExp_{args.weight_dim}", f"Threads_{threads}")
else:
    path = args.path

#  MODEL SELECTION ===========================================================
SMALLEST_LAYER_SIZE_EXPONENT = 7
SMALLEST_LAYER_SIZE = 2**SMALLEST_LAYER_SIZE_EXPONENT
OUTPUT_LAYERS = 1
dim=SMALLEST_LAYER_SIZE
layers = 2**(args.weight_dim-(2*SMALLEST_LAYER_SIZE_EXPONENT+OUTPUT_LAYERS))
if args.verbosity in ("Normal", "Detailed"):
    print(f"layer dimension 2^{SMALLEST_LAYER_SIZE_EXPONENT}")
    print(f"layers {layers}")
    print(f"Pytorch number of threads {torch.get_num_threads()}")


# Data Loader ===============================================================
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, training_set_size, dim):
        'Initialization'
        self.X = torch.rand(training_set_size, dim)
        self.y = torch.rand(args.trainingSetSize, dim)

  def __len__(self):
        'Denotes the toal number of samples'
        return len(self.X)

  def __getitem__(self, idx):
        'Generates one sample of data'
        # Select sample
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load data and get label
        X = self.X[idx]
        y = self.y[idx]

        return X, y

# Model Definition ==========================================================
class LinBench(nn.Module):
    def __init__(self, layers, layer_size):
        super(LinBench, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Linear(layer_size, layer_size))
            self.layers.append(nn.ReLU(True))


    def forward(self, x):
        # Apply convolutions
        for layer in self.layers:
            x = layer(x)
        return x

# test = LinJitBench(layers, SMALLEST_LAYER_SIZE)
test = LinBench(layers, SMALLEST_LAYER_SIZE)
test = torch.jit.script(test)
# Loss Function and Optimizer ===============================================
loss_fn = torch.nn.MSELoss()
params_to_optimize = [
    {'params': test.parameters()},
]
optimizer = torch.optim.Adam(params_to_optimize, lr=args.initialLearningRate)
# Device definition =========================================================
use_cuda = False
if args.engine == "CuDNN":
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cpu")
test.to(device)
# Create Datasets ===========================================================
# Parameters
loader_args = {
        'batch_size': args.trainingBS,
        'shuffle': False,
        'num_workers': 1,
        'pin_memory': False
     }

# Generators
validation_set = Dataset(args.trainingSetSize, dim)
data_generator = torch.utils.data.DataLoader(validation_set, **loader_args)

# Training Loop Definition ==================================================
def train_epoch(nn, data_generator, device, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    nn.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for Xmini, ymini in data_generator:
        # Move tensor to the proper device
        Xmini_dev, ymini_dev = Xmini.to(device), ymini.to(device)

        # Run Forward Pass
        yhat_dev = nn(Xmini_dev)
        # Evaluate loss
        loss = loss_fn(yhat_dev, ymini_dev)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss).item()
# Training Loop =============================================================
if args.submit:
    times = []
    for e_idx, epoch in enumerate(range(args.epochs)):
        # Training
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        tp_start = time.time_ns()
        train_loss = train_epoch(test, data_generator, device, loss_fn, optimizer)
        times.append(time.time_ns()-tp_start)
        if args.verbosity in ("Normal", "Detailed"):
            print(f"Epoch {e_idx} Training Loss {train_loss}")

    runtime_per_epochs = sum(times)/len(times)
    runtime_per_epochs = runtime_per_epochs*10**-9
    print(f"Average Runtime of {runtime_per_epochs}s per Epoch")

    if args.save:
        mkdir_p(path)
        with open(os.path.join(path, "results.csv"), 'w') as fb:
            fb.write(str(runtime_per_epochs))

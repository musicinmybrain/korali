#!/usr/bin/env ipython
import os
import sys
import numpy as np
import korali
from korali.auxiliar.printing import *
import argparse
import cpuinfo
import matplotlib.pyplot as plt
from korali.profiler.helpers import get_total_stats, get_cumulative_results, minimum_duration_to_display, plot_cum, plot_runtimes, plot_all
import seaborn as sns
sys.path.append(os.path.abspath('./_models'))
def mkdir_p(dir):
    """Make a directory if it doesn't exist and create intermediates as well."""
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

k = korali.Engine()
sns.set()

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
    default=2*15,
    type=int,
    required=False)
parser.add_argument(
    '--trainingBS',
    help='Batch size to use for training data',
    type=int,
    default=32,
    # choices=[32, 256, 512 1024, 8192, 16384, 32768],
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
    required=True,
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
    required=True,
    choices=[15,17,19,21,23,25],
    type=int,
)
parser.add_argument(
    '--verbosity',
    help='Verbosity to print',
    default="Normal",
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
np.random.seed(0xC0FFEE)
# In case of iPython need to temporaily set sys.args to [''] in order to parse them
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
args = parser.parse_args()
sys.argv = tmp

if args.verbosity in ["Normal", "Detailed"]:
    print_header('Korali', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)

e = korali.Experiment()
e["Random Seed"] = 0xC0FFEE
#  MODEL SELECTION ===========================================================
SMALLEST_LAYER_SIZE_EXPONENT = 7
SMALLEST_LAYER_SIZE = 2**SMALLEST_LAYER_SIZE_EXPONENT
if args.model == "single":
    from models import single_layer_ffnn as nn
    layers = 1
    assert (args.weight_dim-1) % 2 == 0
    exponent = ((args.weight_dim-1)/2)
    dim=int(2**exponent)
    e["Problem"]["Input"]["Size"] = dim
    e["Problem"]["Solution"]["Size"] = dim
    if args.verbosity in ("Normal", "Detailed"):
        print(f"layers {layers}")
        print(f"layer dimension 2^{exponent}")
elif args.model == "multi":
    from models import multi_layer_ffnn as nn
    OUTPUT_LAYERS = 1
    dim=SMALLEST_LAYER_SIZE
    layers = 2**(args.weight_dim-(2*SMALLEST_LAYER_SIZE_EXPONENT+OUTPUT_LAYERS))
    e["Problem"]["Input"]["Size"] = dim
    e["Problem"]["Solution"]["Size"] = dim
    if args.verbosity in ("Normal", "Detailed"):
        print(f"layer dimension 2^{SMALLEST_LAYER_SIZE_EXPONENT}")
        print(f"layers {layers}")
else:
    sys.exit(f"{args.model} is not a valid model.")
# ============================================================================
if not args.path:
    path = os.path.join("_korali_result", args.other, f"model_{args.model}", f"BS_{args.trainingBS}", f"WeightExp_{args.weight_dim}", f"Threads_{threads}")
else:
    path = args.path
if args.submit:
    # Create Training Data =====================================================================
    X_train = np.random.rand(args.trainingSetSize, 1, dim).tolist()
    y_train = np.random.rand(args.trainingSetSize, dim).tolist()
    # ==========================================================================================
    ### Loading previous models
    if args.load_model:
        args.validationSplit = 0.0
        e.loadState(os.path.join(path, "latest"))

    if args.verbosity in ("Normal", "Detailed"):
        print(f"FLOPS: {get_flops(args.trainingSetSize, args.trainingBS, dim, layers)}")
    e["Problem"]["Description"] = f"FLOPS_{get_flops(args.trainingSetSize, args.trainingBS, dim, dim)}_THREADS_{threads}"
    e["Solver"]["Batch Concurrency"] = 1
    e["Problem"]["Type"] = "Supervised Learning"
    e["Problem"]["Max Timesteps"] = 1
    e["Problem"]["Training Batch Size"] = args.trainingBS

    e["Problem"]["Input"]["Data"] = X_train
    e["Problem"]["Solution"]["Data"] = y_train


    e["Solver"]["Type"] = "Learner/DeepSupervisor"
    e["Solver"]["Mode"] = "Automatic Training"
    e["Solver"]["Loss Function"] = "Mean Squared Error"

    e["Solver"]["Learning Rate"] = args.initialLearningRate

    e["Solver"]["Data"]["Input"]["Shuffel"] = False
    e["Solver"]["Data"]["Training"]["Shuffel"] = False
    e["Solver"]["Neural Network"]["Engine"] = args.engine
    e["Solver"]["Optimizer"]["Type"] = "learner/deepSupervisor/optimizers/f"+args.optimizer
    # Set Model ================================================================================
    nn(e, dim, layers, SMALLEST_LAYER_SIZE)
    # ==========================================================================================
    # e["Console Output"]["Frequency"] = 0
    e["Console Output"]["Verbosity"] = args.verbosity
    e["File Output"]["Enabled"] = args.save
    e["File Output"]["Frequency"] = 0
    e["File Output"]["Path"] = path
    e["Save Only"] = ["Current Generation", "Run ID", "Time Stamps"]
    e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
    k["Conduit"]["Type"] = "Sequential"
    k.run(e)
flop = get_flops(args.trainingSetSize, args.trainingBS, dim, layers)
# cpu = cpuinfo.get_cpu_info()["brand_raw"]
cpu = "Intel(R) Xeson(R) CPU E5-2695 v4 @ 2.10GHz"
extra_stats = {
    "Threads": threads,
    "Weights Exponents": args.weight_dim,
    "BS": args.trainingBS,
    "CPU": cpu
}
print(f"Runtime {e['Time Stamps'][0]['Generation'][0][1]}")

if args.plot:
    results = get_total_stats(os.path.join(path, "latest"), {})
    cum = get_cumulative_results(results)
    plt.rcParams['figure.figsize'] = 18, 12
    plot_all(results, cum, title = f"{cpu} with {threads} threads, Batch Size {args.trainingBS}, Weights 2^{args.weight_dim} and {layers} layers")
    plt.savefig(os.path.join(path, "profiling.pdf"), dpi = 200)
    plt.show()

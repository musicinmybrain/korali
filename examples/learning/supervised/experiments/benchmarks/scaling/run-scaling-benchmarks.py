#!/usr/bin/env ipython
import os
import sys
import numpy as np
import time
import korali
from korali.auxiliar.printing import *
import math
import argparse
import json
import pandas as pd
import cpuinfo
from collections import defaultdict
sys.path.append(os.path.abspath('./_models'))
def get_profiling_stats_for_each_fun(timings):
    durations = [t[1] for t in timings]
    return {
        "calls" : len(durations),
        "runtime" : sum(durations),
        "means" : np.mean(durations),
        "var" : np.var(durations)
    }

def calc_profiling_stats(results_per_gen, additional):
    results = defaultdict(dict)
    for func_name, func_stats in results_per_gen.items():
        calls_per_gen = [gen["calls"] for gen in func_stats]
        runtime = [gen["runtime"] for gen in func_stats]
        results[func_name] = {
            "Runtime" : sum(runtime),
            "\u03BC runtime [per gen]": np.mean(runtime),
            "\u03C3 runtime [per gen]": math.sqrt(np.mean([gen["var"] for gen in func_stats])),
            "Total #calls": sum(calls_per_gen),
            "\u03BC #calls [per gen]": np.mean(calls_per_gen),
            "\u03C3 #calls [per gen]": np.std(calls_per_gen)
        }
        results[func_name].update(additional)
    return results

def get_results_per_gen(timings):
    results_per_gen = defaultdict(list)
    for timings_per_gen in timings:
        for function_name, f_timings in timings_per_gen.items():
            results_per_gen[function_name].append(get_profiling_stats_for_each_fun(f_timings))
    return results_per_gen

def create_data_frame(profiling_stats):
    list_of_dicts = []
    for func_name, values in profiling_stats.items():
        values["Function"] = func_name
        list_of_dicts.append(values)
    results = pd.DataFrame.from_records(list_of_dicts, index = "Function")
    results.insert(loc=0, column="% of Gen Time", value = (results["Runtime"]/results.loc["_solver->runGeneration", "Runtime"]*100).astype(int))
    results = results.sort_values('Runtime', ascending=False)
    return results

def get_profiling_stats(path, additional):
    with open(path) as fb:
        data = json.load(fb)
        epochs = data["Solver"]["Epoch Count"]
        timings = data["Time Stamps"]
    results_per_gen = get_results_per_gen(timings)
    data = calc_profiling_stats(results_per_gen, additional)
    return create_data_frame(data)

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
    default=3,
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
    choices=[32, 1024],
    required=False)
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
    path = os.path.join("_korali_result", f"weights_{args.weight_dim}", args.other, args.model)
else:
    path = args.path

# Create Training Data =====================================================================
X_train = np.random.rand(args.trainingSetSize, 1, dim).tolist()
y_train = np.random.rand(args.trainingSetSize, dim).tolist()
# ==========================================================================================
### Loading previous models
if args.load_model:
    args.validationSplit = 0.0
    e.loadState(os.path.join(path, "latest"))

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

if args.verbosity in ("Normal", "Detailed"):
    print(f"FLOPS: {get_flops(args.trainingSetSize, args.trainingBS, dim, layers)}")
e["Problem"]["Description"] = f"FLOPS_{get_flops(args.trainingSetSize, args.trainingBS, dim, dim)}_THREADS_{args.threads}"
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
# e["Save Only"] = ["Current Generation" ,"Run ID", "Solver"]
e["Solver"]["Termination Criteria"]["Epochs"] = args.epochs
k["Conduit"]["Type"] = "Sequential"
k.run(e)
flop = get_flops(args.trainingSetSize, args.trainingBS, dim, layers)
extra_stats = {
    "Threads": args.threads,
    "Flops": flop,
    "Weights Exponents": args.weight_dim,
    "BS": args.trainingBS
}
df = get_profiling_stats(os.path.join(path, "latest"), extra_stats)
df.to_csv(os.path.join(path, f"results.csv"), sep=',', index=False, float_format='%.3f')
info = cpuinfo.get_cpu_info()["brand_raw"]
print(f'Run on CPU: {info}')
print(df)

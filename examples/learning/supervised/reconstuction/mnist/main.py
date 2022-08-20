#!/usr/bin/env ipython
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
import json
import matplotlib
matplotlib.use('TkAgg')
from korali.auxiliar.printing import *
sys.path.append(os.path.abspath('./_models'))
sys.path.append(os.path.abspath('..'))
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from korali.plot.helpers import hlsColors, drawMulticoloredLine
import seaborn as sns
from utilities import make_parser
from utilities import pcolors

def plot_loss(results):
    # ax.set_yscale(args.yscale)
    pc = pcolors()
    x = range(1, results["Epoch"]+1)
    ax.semilogy(x, results["Training Loss"] ,results["style"], label="Training Loss"+results["type"], color=pc.train)
    if 'Validation Loss' in results:
        ax.semilogy(x, results["Validation Loss"] ,results["style"], label="Validation Loss"+results["type"], color=pc.val)
        ax.set_xlabel('Epochs')
    ylabel = results['Loss Function']
    if "Regularizer" in results:
        ylabel+= " + " + results["Regularizer"]["Type"]
        ax.set_ylabel(ylabel)
    plt.legend()
def plot_timings(results):
    values = {v["Type"]: v["Time per Epoch"] for (k, v) in results.items()}
    plt.bar(values.keys(), values.values())

line_styles = ["-", "--", "-.", "."]

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    # results_files = ["_korali_result_automatic", "_korali_result", "pytorch/result"]
    model = "linear"
    results_files = {
        "Korali Automatic": "_korali_result_automatic",
        "Korali": "_result_automatic",
        "Pytorch": "pytorch/result"
    }
    results_files = {k: os.path.join(cwd, f, model, "latest") for k, f in results_files.items()}
    results = []
    for idx, (k, f) in enumerate(results_files.items()):
        with open(f, 'r') as fs:
            result = json.load(fs)
            if "Results" in result:
                result = result["Results"]
            result["style"] = line_styles[idx]
            result["style"] = line_styles[idx]
            result["Type"] = k
            results.append(result)
# Plot Loses ===================================================================
    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Autoencoder")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    for r in results:
        plot_loss(r)
        # if 'Learning Rate' in results:
        #     ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        #     ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
        #     ax2.plot(epochs, results["Learning Rate"], color=lr_c)
        #     fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #
# Plot Timings =================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Autoencoder")
    plot_timings(results)
    plt.show()

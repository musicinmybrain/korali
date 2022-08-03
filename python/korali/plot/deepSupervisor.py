#! /usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from korali.plot.helpers import hlsColors, drawMulticoloredLine
import seaborn as sns
sns.set()
palette_tab10 = sns.color_palette("tab10", 10)
plt.rcParams['text.usetex'] = True
dl_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
dl_parser.add_argument(
  "--yscale",
  help="yscale to plot",
  default="log",
  required=False,
  choices=["linear", "log"]
)

# Plot CMAES results (read from .json files)
def plot(gen_dicts, config, others, **kwargs):
  args = dl_parser.parse_args(others)
  gen_dicts.popitem()
  for gen, gen_dict in gen_dicts.items():
    if "Mode" not in gen_dict["Results"]:
      sys.exit(f'The file of generation {gen} does not contain the the field ["Results"]["Mode"]')
    if gen_dict["Results"]["Mode"] != 'Predict':
      gen_dicts[gen] = gen_dict

  gens = len(gen_dicts)
  gen_numbers = [gen for gen in gen_dicts]
  FIRST_GEN = min(gen_numbers)
  EPOCHS = max(gen_numbers)
  # config = gen_dicts[list(gen_dicts)[-1]]
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
  TrainingLoss = [gen["Results"]["Training Loss"] for gen in gen_dicts.values()]
  ax.plot(gen_dicts.keys(), TrainingLoss ,'-', label="Training Loss", color=palette_tab10[0])
  ax.set_yscale(args.yscale)
  if 'Validation Loss' in config['Results']:
    ValidationLoss = [gen["Results"]["Validation Loss"] for gen in gen_dicts.values()]
    ax.plot(gen_dicts.keys(), ValidationLoss ,'-', label="Validation Loss", color=palette_tab10[1])
  ax.set_xlabel('Epochs')
  ylabel = config['Results']['Loss Function']
  if "Regularizer" in config["Results"]:
    ylabel+= " + " + config["Results"]["Regularizer"]["Type"]
  ax.set_ylabel(ylabel)
  if "Description" in config["Results"]:
    ax.set_title(config["Results"]["Description"])
  plt.legend()
  if 'Learning Rate' in config['Results']:
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
    ax2.plot(gen_dicts.keys(), [gen["Results"]["Learning Rate"] for gen in gen_dicts.values()], color=palette_tab10[4])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

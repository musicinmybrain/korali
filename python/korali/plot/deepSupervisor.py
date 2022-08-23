#! /usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from korali.plot.helpers import hlsColors, drawMulticoloredLine
import seaborn as sns
sns.set()
palette = sns.color_palette("tab10")
train_c = palette[2]
val_c = palette[3]
lr_c = palette[5]
test_c = palette[4]
f_c = palette[0]
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
  gens = []
  for gen, gen_dict in gen_dicts.items():
    if "Mode" not in gen_dict["Results"]:
      sys.exit(f'The file of generation {gen} does not contain the the field ["Results"]["Mode"]')
    if gen_dict["Results"]["Mode"] != 'Predict':
      gens.append(gen_dict)

  last_gen = gens[-1]
  EPOCHS = last_gen["Results"]["Epoch"]
  X = range(1, EPOCHS+1)
  # config = gen_dicts[list(gen_dicts)[-1]]
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
  TrainingLoss = last_gen["Results"]["Training Loss"]
  ax.plot(X, TrainingLoss ,'-', label="Training Loss", color=train_c)
  ax.set_yscale(args.yscale)
  if 'Validation Loss' in config['Results']:
    ValidationLoss = last_gen["Results"]["Validation Loss"]
    ax.plot(X, ValidationLoss ,'-', label="Validation Loss", color=val_c)
  ax.set_xlabel('Epochs')
  ylabel = config['Results']['Loss Function']
  if "Regularizer" in config["Results"]:
    ylabel+= " + " + config["Results"]["Regularizer"]["Type"]
  ax.set_ylabel(ylabel)
  if "Description" in config["Results"]:
    ax.set_title(config["Results"]["Description"].capitalize())
  plt.legend()
  if 'Learning Rate' in config['Results']:
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
    ax2.plot(gen_dicts.keys(), [gen["Results"]["Learning Rate"] for gen in gen_dicts.values()], color=lr_c)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

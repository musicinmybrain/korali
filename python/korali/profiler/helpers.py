import json
import os
import numpy as np
import math
import pandas as pd
from collections import defaultdict
from matplotlib import rcParams
from matplotlib import pyplot as plt
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set()

def get_total_profiling_stats_for_each_fun(gen_idx, f_name, timings, additional = None):
    l_of_function_calls = []
    for call_idx, values in enumerate(timings):
        df = {
                "generation": gen_idx,
                "name": f_name,
                "call": call_idx,
                "start": values[0],
                "end": values[0]+values[1],
                "duration": values[1]
            }
        if additional:
            df = df | additional
        l_of_function_calls.append(df)
    return l_of_function_calls

def get_total_results_per_gen(timings, additional):
    all_functions = []
    for gen_idx, timings_per_gen in enumerate(timings):
        for function_name, f_timings in timings_per_gen.items():
            all_functions.extend(get_total_profiling_stats_for_each_fun(gen_idx, function_name, f_timings, additional))
    return all_functions

def get_total_stats(path, additional):
    with open(path) as fb:
        data = json.load(fb)
        timings = data["Time Stamps"]
    df = pd.DataFrame(get_total_results_per_gen(timings, additional))
    df = df.set_index('name')
    return df

def get_cumulative_results(df):
    cumulative = df.groupby("name").sum()
    cumulative = cumulative.sort_values("duration",  ascending=False)
    cumulative["percentage"] = cumulative["duration"]/cumulative["duration"].max()*100
    return cumulative[["percentage", "duration"]]


def plot_cum(cumulative, fig, gs):
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax, kind in zip([ax1, ax2], ["duration", "percentage"]):
        ax.barh(cumulative.index.tolist(), cumulative[kind].tolist())
        ax.set_title(kind)
        for i, value in enumerate(cumulative[kind].tolist()):
            ax.text(value + 1, i, "{:.1f}".format(value), color = 'black')

def minimum_duration_to_display(row, MIN_DISPLAY_DURATION):
    """
        If val < THRESHOLD use MIN_DISPLAY_DURATION
    """
    if (row["duration"] > 0) and (row["duration"] < MIN_DISPLAY_DURATION):
        return MIN_DISPLAY_DURATION
    else:
        return row["duration"]

def plot_runtimes(df, fig, gs, title="Runtime Plot"):
    total_runtime = df["end"].max()-df["start"].min()
    one_percent_runtime = (total_runtime)/100
    df.duration = df.apply(lambda row : minimum_duration_to_display(row, 0.15*one_percent_runtime), axis=1)
    palette = sns.color_palette("deep")
    ax = fig.add_subplot(gs[1, :])

    VERTICAL_OFFSET_BETWEEN_BARS = 2
    BAR_HEIGHT = 10
    ymin = 0
    yticks = []
    ylabels = []
    for idx, func_name in enumerate(df.index.unique()):
        # Horizontal sequence of rectangles
        for gen in df["generation"].unique():
            df_name = df.loc[(df.index == func_name) & (df["generation"] == gen)]
            xminmax = list(zip(df_name["start"], df_name["duration"]))
            ymin = idx*(BAR_HEIGHT+VERTICAL_OFFSET_BETWEEN_BARS)
            ax.broken_barh(xminmax, (ymin, BAR_HEIGHT), facecolors=palette[gen % len(palette)])
        calls = df.groupby("name").max()["call"].loc[func_name]
        yticks.append(ymin+BAR_HEIGHT/2)
        ylabels.append(f"{func_name}\n {calls} calls")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_title(title)
    # ylim and xlim of the axes
    ax.set_ylim(0, ymin+BAR_HEIGHT)
    ax.set_xlim(df["start"].min(), df["end"].max()+one_percent_runtime);

def plot_all(results, cum, title = ""):
    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    plot_cum(cum, fig, gs)
    # plot runtimes
    plot_runtimes(results, fig, gs)
    total_runtime = results.groupby("name").max()["duration"].loc["Generation"]
    title += f" runtime {total_runtime:.2f}s"
    fig.suptitle(title)

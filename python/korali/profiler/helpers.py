import json
import os
import numpy as np
import math
import pandas as pd
from collections import defaultdict
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set()


def get_profiling_stats_for_each_fun(timings):
    start_times = [t[0] for t in timings]
    durations = [t[1] for t in timings]
    return {
        "calls" : len(durations),
        "runtime" : sum(durations),
        "means" : np.mean(durations),
        "var" : np.var(durations)
    }

def calc_profiling_stats(results_per_gen):
    results = defaultdict(dict)
    for func_name, func_stats in results_per_gen.items():
        calls_per_gen = [gen["calls"] for gen in func_stats]
        runtime = [gen["runtime"] for gen in func_stats]
        results[func_name] = {
            "Runtime" : sum(runtime),
            "\u03BC runtime [per gen]" : np.mean(runtime),
            "\u03C3 runtime [per gen]" : math.sqrt(np.mean([gen["var"] for gen in func_stats])),
            "Total #calls" : sum(calls_per_gen),
            "\u03BC #calls [per gen]" : np.mean(calls_per_gen),
            "\u03C3 #calls [per gen]" : np.std(calls_per_gen)
        }
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

def get_profiling_stats(path):
    with open(path) as fb:
        data = json.load(fb)
        epochs = data["Solver"]["Epoch Count"]
        timings = data["Time Stamps"]
    results_per_gen = get_results_per_gen(timings)
    data = calc_profiling_stats(results_per_gen)
    return create_data_frame(data)

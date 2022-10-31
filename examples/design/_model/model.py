import numpy as np
import sys

def model(s):
    theta = np.array(s["Parameters"])
    d = np.array(s["Designs"])
    res = theta * theta * theta * d * d + theta * np.exp(-np.abs(0.2 - d))
    s["Model Evaluation"] = res.tolist()
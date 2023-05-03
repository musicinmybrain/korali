#!/usr/bin/env python3

# In this example, TODO

import sys
sys.path.append('./_model')

import matplotlib.pyplot as plt
import numpy as np
from model import solveLinearODEs

import argparse
parser = argparse.ArgumentParser(
                    prog='makeSyntheticData.py',
                    description='Make dnoisy date from ODE model and plot')
parser.add_argument('--N', help='Number of data.', default=10, type=int)
parser.add_argument('--noise', help='Standard deviation of Gaussian noise', default=1.0, type=float)

args = parser.parse_args()

IC = [2, 10]
theta = [0.3, 0.7]
tEval = np.linspace(0, 20, num=args.N)

out = solveLinearODEs(theta, IC, tEval)
print(out)

plt.plot(tEval, out)
plt.savefig("data.png")


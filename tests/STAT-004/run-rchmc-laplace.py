#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
for useDiagonalMetric in [False, True]:
    k = korali.Engine()
    e = korali.Experiment()

    # Selecting problem and solver types.
    e["Problem"]["Type"] = "Sampling"
    e["Problem"]["Probability Function"] = llaplace

    # Defining problem's variables and their HMC settings
    e["Variables"][0]["Name"] = "X0"
    e["Variables"][0]["Initial Mean"] = 0.0
    e["Variables"][0]["Initial Standard Deviation"] = 1.0

    # Configuring the HMC sampler parameters
    e["Solver"]["Type"] = "Sampler/HMC"
    e["Solver"]["Burn In"] = 500
    e["Solver"]["Termination Criteria"]["Max Samples"] = 100000

    # HMC specific parameters
    e["Solver"]["Num Integration Steps"] = 20
    e["Solver"]["Step Size"] = 0.05
    e["Solver"]["Version"] = 'Riemannian Const'
    e["Solver"]["Use Diagonal Metric"] = useDiagonalMetric
    e["Solver"]["Use Adaptive Step Size"] = True
    e["Solver"]["Target Integration Time"] = 0.5
    e["Solver"]["Target Acceptance Rate"] = 0.80
    e["Solver"]["Use NUTS"] = False

    e["Console Output"]["Frequency"] = 5000
    e["File Output"]["Frequency"] = 0
    e["File Output"]["Enabled"] = False

    # Running Korali
    e["Random Seed"] = 1227
    k.run(e)

    verifyMean(e["Solver"]["Sample Database"], [4.0], 0.1)
    verifyStd(e["Solver"]["Sample Database"], [math.sqrt(2)], 0.05)

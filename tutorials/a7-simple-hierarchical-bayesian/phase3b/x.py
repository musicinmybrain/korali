#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Creating hierarchical Bayesian problem from previous two problems
e = korali.newExperiment()

e["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/Theta"
e["Problem"]["Theta Problem Path"] = '../setup/results_phase_1/000'
e["Problem"]["Psi Problem Path"] = '../setup/results_phase_2'

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 1000
e["Solver"]["Default Burn In"] = 2;
e["Solver"]["Max Chain Length"] = 1;
e["Solver"]["Target Coefficient Of Variation"] = 0.6

e["Verbosity"] = "Detailed"
e["Result Path"] = "../setup/results_phase_3b/"

# Starting Korali's Engine and running experiment
k = korali.initialize()
k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)

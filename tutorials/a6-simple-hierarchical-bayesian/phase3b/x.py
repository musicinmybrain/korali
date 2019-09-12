#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali
sys.path.append('../setup/model')
from model import *

# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"]  = "Hierarchical Bayesian (Theta)"

kH["Problem"]["Sub Problem"] = '../setup/results_phase_1/000/final.json'
kH["Problem"]["Psi Problem"] = '../setup/results_phase_2/final.json'

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 2000
kH["Solver"]["Default Burn In"] = 2;
kH["Solver"]["Max Chain Length"] = 1;
kH["Solver"]["Target Coefficient Of Variation"] = 0.6

kH["General"]["Console Output"]["Verbosity"] = "Detailed"
kH["General"]["Results Output"]["Path"] = "../setup/results_phase_3b/"


kH.run()

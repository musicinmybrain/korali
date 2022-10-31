#!/usr/bin/env python3
import os
import sys
import math
sys.path.append('./model/')
from model import *
import numpy as np

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Integration"
e["Problem"]["Integrand"] = pconstant

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0
e["Variables"][0]["Number Of Gridpoints"] = 100

e["Solver"]["Type"] = "Integrator/Quadrature"
e["Solver"]["Method"] = "Rectangle"
e["Solver"]["Executions Per Generation"] = 100

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
  
# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)


# Verify result
assert (math.isclose(e["Results"]["Integral"], 1., rel_tol=0., abs_tol=1e-6))

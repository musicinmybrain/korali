#!/usr/bin/env python3
import os
import sys
sys.path.append('./model/')
from model import model_integration
import numpy as np

### With Predetermined values

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Integration"
e["Problem"]["Integrand"] = model_integration

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0

e["Solver"]["Type"] = "Integrator/MonteCarlo"
e["Solver"]["Number Of Samples"] = 100
e["Solver"]["Executions Per Generation"] = 10

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
  
# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

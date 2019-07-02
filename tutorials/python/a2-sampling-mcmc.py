#!/usr/bin/env python3

## In this example, we demonstrate how Korali samples the posterior
## distribution in a bayesian problem where the likelihood
## is provided directly by the computational model.
## In this case, we use the MCMC method.

# Importing computational model
import sys
sys.path.append('./model')
from directModel import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Setting Model
k.setModel(evaluateModel)

# Selecting problem and solver types.
k["Problem"] = "Direct Evaluation"
k["Solver"] = "MCMC"

# Defining problem's variables and their prior distribution
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["MCMC"]["Initial Mean"] = 0.0
k["Variables"][0]["MCMC"]["Standard Deviation"] = 1.000

# Configuring the MCMC sampler parameters
k["MCMC"]["Burn In"] = 500
k["MCMC"]["Chain Length"]  = 5000
k["MCMC"]["Adaptive Sampling"]  = True
k["MCMC"]["Result Output Frequency"]  = 5000

# Setting output directory
k["Result Directory"] = "a2_sampling_mcmc_result"

# Running Korali
k.run()

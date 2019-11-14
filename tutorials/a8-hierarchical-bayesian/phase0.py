#!/usr/bin/env python3
import sys
import numpy as np
sys.path.append('setup/model')
from model import *

N = 5;

import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Execution/Model"
e["Problem"]["Execution Model"] = lambda modelData: logistic_reference(modelData)

e["Variables"][0]["Name"] = "V1"
e["Variables"][1]["Name"] = "V2"
e["Variables"][2]["Name"] = "V3"
e["Variables"][3]["Name"] = "V4"

e["Variables"][0]["Loaded Values"] = (300*np.ones(N)).tolist()
e["Variables"][1]["Loaded Values"] = np.random.normal( 40, 4, N ).tolist()
e["Variables"][2]["Loaded Values"] = np.random.lognormal( 0, 0.5, N ).tolist()
e["Variables"][3]["Loaded Values"] = (5*np.ones(N)).tolist()

e["Solver"]["Type"] = "Executor"
e["Result Path"] = "setup/results_phase_0/"
e["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

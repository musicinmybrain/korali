#!/usr/bin/env python3
import math
import sys
sys.path.append('./_model/')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Design"
e["Problem"]["Model"] = model

e["Solver"]["Type"] = "Designer"
numMeasurements = 1
e["Solver"]["Sigma"] = 1e-2

e["Distributions"][0]["Name"] = "Uniform"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = 1.0

indx = 0
e["Variables"][indx]["Name"] = "d1"
e["Variables"][indx]["Type"] = "Design"
e["Variables"][indx]["Number Of Samples"] = 101
e["Variables"][indx]["Lower Bound"] = 0.0
e["Variables"][indx]["Upper Bound"] = 1.0
e["Variables"][indx]["Distribution"] = "Grid"
indx += 1

if numMeasurements == 2:
	e["Variables"][indx]["Name"] = "d2"
	e["Variables"][indx]["Type"] = "Design"
	e["Variables"][indx]["Number Of Samples"] = 101
	e["Variables"][indx]["Lower Bound"] = 0.0
	e["Variables"][indx]["Upper Bound"] = 1.0
	e["Variables"][indx]["Distribution"] = "Grid"
	indx += 1

e["Variables"][indx]["Name"] = "theta"
e["Variables"][indx]["Type"] = "Parameter"
e["Variables"][indx]["Lower Bound"] = 0.0
e["Variables"][indx]["Upper Bound"] = 1.0
e["Variables"][indx]["Number Of Samples"] = 1e2
e["Variables"][indx]["Distribution"] = "Uniform"
indx += 1

e["Variables"][indx]["Name"] = "y1"
e["Variables"][indx]["Type"] = "Measurement"
e["Variables"][indx]["Number Of Samples"] = 1e2
indx += 1

if numMeasurements == 2:
	e["Variables"][indx]["Name"] = "y2"
	e["Variables"][indx]["Type"] = "Measurement"
	e["Variables"][indx]["Number Of Samples"] = 1e2

# Starting Korali's Engine and running experiment
k = korali.Engine()
# k["Conduit"]["Type"] = "Concurrent"
# k["Conduit"]["Concurrent Jobs"] = 12
k.run(e)

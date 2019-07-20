#!/usr/bin/env python3
import korali
import sys
sys.path.append("model")
from runModel import *

k = korali.Engine()

k["Problem"]["Type"] = "Optimization";
k["Problem"]["Objective"] = "Maximize"

k["Variables"][0]["Name"] = "X0"
k["Variables"][0]["Lower Bound"] = -32.0;
k["Variables"][0]["Upper Bound"] = +32.0;

k["Variables"][1]["Name"] = "X1"
k["Variables"][1]["Lower Bound"] = -32.0;
k["Variables"][1]["Upper Bound"] = +32.0;

k["Variables"][2]["Name"] = "X2"
k["Variables"][2]["Lower Bound"] = -32.0;
k["Variables"][2]["Upper Bound"] = +32.0;

k["Variables"][3]["Name"] = "X3"
k["Variables"][3]["Lower Bound"] = -32.0;
k["Variables"][3]["Upper Bound"] = +32.0;
  
k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = int(sys.argv[1])

k["Solver"]["Type"] = "CMAES"
k["Solver"]["Sample Count"] = 10
  
k["General"]["Max Generations"] = 30

k.setModel(runModel)
k.run()

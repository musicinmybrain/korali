#!/usr/bin/env python3

## In this example, we demonstrate how Korali simulates a reaction.

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()
e["Random Seed"] = 0xC001

# Configuring Problem
e["Problem"]["Type"] = "Reaction"
e["Problem"]["Reactions"][0]["Equation"] = "[X1]->Y1"
e["Problem"]["Reactions"][0]["Rate"] = 0.1

e["Problem"]["Reactions"][1]["Equation"] = "[X2]+Y1->Y2+Z1"
e["Problem"]["Reactions"][1]["Rate"] = 0.1

e["Problem"]["Reactions"][2]["Equation"] = "2 Y1+Y2->3 Y1"
e["Problem"]["Reactions"][2]["Rate"] = 5e-5

e["Problem"]["Reactions"][3]["Equation"] = "Y1->Z2"
e["Problem"]["Reactions"][3]["Rate"] = 5.

# Defining Reactants
e["Variables"][0]["Name"] = "[X1]"
e["Variables"][0]["Initial Reactant Number"] = 50000

e["Variables"][1]["Name"] = "[X2]"
e["Variables"][1]["Initial Reactant Number"] = 500

e["Variables"][2]["Name"] = "Y1"
e["Variables"][2]["Initial Reactant Number"] = 1000

e["Variables"][3]["Name"] = "Y2"
e["Variables"][3]["Initial Reactant Number"] = 2000

e["Variables"][4]["Name"] = "Z1"
e["Variables"][4]["Initial Reactant Number"] = 0

e["Variables"][5]["Name"] = "Z2"
e["Variables"][5]["Initial Reactant Number"] = 0

# Configuring SSA parameters
e["Solver"]["Type"] = "SSM/SSA"
e["Solver"]["Simulations Per Generation"] = 1
e["Solver"]["Simulation Length"] = 14
e["Solver"]["Termination Criteria"]["Max Num Simulations"] = 1
e["Solver"]["Diagnostics"]["Num Bins"] = 500

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_brusselator_ssa'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)

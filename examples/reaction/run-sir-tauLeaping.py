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
e["Problem"]["Reactions"][0]["Equation"] = "S+I->2I"
e["Problem"]["Reactions"][0]["Rate"] = 0.0005
e["Problem"]["Reactions"][1]["Equation"] = "I->R"
e["Problem"]["Reactions"][1]["Rate"] = 0.2

# Configuring Reactants
e["Variables"][0]["Name"] = "S"
e["Variables"][0]["Initial Reactant Number"] = 5000

e["Variables"][1]["Name"] = "I"
e["Variables"][1]["Initial Reactant Number"] = 5

e["Variables"][2]["Name"] = "R"
e["Variables"][2]["Initial Reactant Number"] = 0

# Configuring TauLEaping parameters
e["Solver"]["Type"] = "SSM/TauLeaping"
e["Solver"]["Simulation Length"] = 20.
e["Solver"]["Simulations Per Generation"] = 100
e["Solver"]["Nc"] = 100
e["Solver"]["Epsilon"] = 0.03
e["Solver"]["Num SSA Steps"] = 100
e["Solver"]["Acceptance Factor"] = 10
e["Solver"]["Termination Criteria"]["Max Num Simulations"] = 1000
e["Solver"]["Diagnostics"]["Num Bins"] = 500

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_sir_tau_leaping'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)

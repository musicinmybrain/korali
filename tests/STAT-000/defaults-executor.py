#!/usr/bin/env python3
import korali

k = korali.initialize()

k["Problem"]["Type"] = "Execution"
k["Problem"]["Execution Model"] = lambda x : x

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Loaded Values"] = []

k["Solver"]["Type"] = "Executor"
#k["Solver"]["Executions Per Generation"] = 100

#k["Console Output"]["Verbosity"] = "Detailed"

k.dry()

###############################################################################

# Testing Configuration

assert k["Solver"]["Executions Per Generation"] == 0

# Testing Internals

assert k["Solver"]["Internal"]["Execution Count"] == 0
assert k["Solver"]["Internal"]["Sample Count"] == 0
assert k["Solver"]["Internal"]["Variable Count"] == 1

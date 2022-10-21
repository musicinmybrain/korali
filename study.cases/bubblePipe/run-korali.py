#!/usr/bin/env python3
import korali
import sys
import argparse
sys.path.append("_model/")
from model import *

k = korali.Engine()
e = korali.Experiment()

# Parsing arguments
parser = argparse.ArgumentParser(description='Runs the Aphros Fishbone experiment with Korali.')
parser.add_argument('--resultFolder', '-r', help='Path to the resuls folder', default='_result', required=False)
parser.add_argument('--samples', '-s', help='Number of CMAES samples per generation.', default=32, required=False)
parser.add_argument('--concurrency', '-c', help='Number of concurrent Aphros instances.', default=1, required=False)
parser.add_argument('--objective', '-obj', help='Optimization objective', choices=['maxNumCoal', 'minNumCoal', 'maxMeanVel'],   required=True)
parser.add_argument('--ngens', '-ng', help='Number of generations to run per job.', default=200, required=False)
args = parser.parse_args()

# Parsing inputs
objective = str(args.objective)
ngens = int(args.ngens)

# Loading previous results if they exist
resultFolder = args.resultFolder
e["File Output"]["Path"] = resultFolder
found = e.loadState(resultFolder + '/latest')

# If, found adding number of generations to the termination criteria
if (found == True):
 e["Solver"]["Termination Criteria"]["Max Generations"] = e["Current Generation"] + ngens
else:
 e["Solver"]["Termination Criteria"]["Max Generations"] = ngens
 
# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = lambda x: model(x, resultFolder, objective)

e["Variables"][0]["Name"] = "Arc Width 1"
e["Variables"][0]["Lower Bound"] = 0.5
e["Variables"][0]["Upper Bound"] = 1.5
e["Variables"][0]["Initial Standard Deviation"] = 0.5

e["Variables"][1]["Name"] = "Arc Width 1"
e["Variables"][1]["Lower Bound"] = 0.5
e["Variables"][1]["Upper Bound"] = 1.5
e["Variables"][1]["Initial Standard Deviation"] = 0.5

e["Variables"][2]["Name"] = "Arc Width 2"
e["Variables"][2]["Lower Bound"] = 0.5
e["Variables"][2]["Upper Bound"] = 1.5
e["Variables"][2]["Initial Standard Deviation"] = 0.5

e["Variables"][3]["Name"] = "Arc Width 3"
e["Variables"][3]["Lower Bound"] = 0.5
e["Variables"][3]["Upper Bound"] = 1.5
e["Variables"][3]["Initial Standard Deviation"] = 0.5

e["Variables"][4]["Name"] = "Arc Width 4"
e["Variables"][4]["Lower Bound"] = 0.5
e["Variables"][4]["Upper Bound"] = 1.5
e["Variables"][4]["Initial Standard Deviation"] = 0.5

e["Variables"][5]["Name"] = "Arc Offset 1"
e["Variables"][5]["Lower Bound"] = -0.5
e["Variables"][5]["Upper Bound"] = 0.5
e["Variables"][5]["Initial Standard Deviation"] = 0.5

e["Variables"][6]["Name"] = "Arc Offset 2"
e["Variables"][6]["Lower Bound"] = -0.5
e["Variables"][6]["Upper Bound"] = 0.5
e["Variables"][6]["Initial Standard Deviation"] = 0.5

e["Variables"][7]["Name"] = "Arc Offset 3"
e["Variables"][7]["Lower Bound"] = -0.5
e["Variables"][7]["Upper Bound"] = 0.5
e["Variables"][7]["Initial Standard Deviation"] = 0.5

e["Variables"][8]["Name"] = "Arc Offset 4"
e["Variables"][8]["Lower Bound"] = -0.5
e["Variables"][8]["Upper Bound"] = 0.5
e["Variables"][8]["Initial Standard Deviation"] = 0.5

e["Variables"][9]["Name"] = "Arc Offset 5"
e["Variables"][9]["Lower Bound"] = -0.5
e["Variables"][9]["Upper Bound"] = 0.5
e["Variables"][9]["Initial Standard Deviation"] = 0.5

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = args.samples

# Result Settings
e["File Output"]["Path"] = resultFolder
e["File Output"]["Frequency"] = 1 # Saving every state

# Selecting external conduit
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = int(args.concurrency)

# Reproducibility Settings
e["Random Seed"] = 0xC0FFEE
e["Preserve Random Number Generator States"] = True

# Configuring base parameter file
parameterString = ''

# Logging configuration
print('--------------------------------------------------')
print('Running Korali+Aphros Pipe experiment.')
print('Result Folder: ' + resultFolder)
print('Generations per job: ' + str(args.ngens))
print('CMAES samples per generation: ' + str(args.samples))
print('Concurrent Aphros instances: ' + str(args.concurrency))
print('Objective: ' + objective)
print('Base Configuration:')
print(parameterString, end='')
print('--------------------------------------------------')

# Storing base parameter file
configFile='_config/par.py'
with open(configFile, 'w') as f:
  print('[Korali] Creating: ' + configFile + '...')
  f.write(parameterString)
  
k.run(e)

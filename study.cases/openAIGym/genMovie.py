#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--input', help='Specifies the result folder to load the best policy from.', required=True)
parser.add_argument('--output', help='Specifies the output folder for the movie.', required=True)
args = parser.parse_args()

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results

initEnvironment(e, args.env, args.output)
found = e.loadState(args.input  + '/latest');

if (found == False): 
 print('Error: could not find results in folder: ' + args.input)
 exit(-1)

### Enabling step information output

e["Problem"]["Custom Settings"]["Print Step Information"] = "Enabled"

### Setting file output configuration

e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = [ 0 ]
e["File Output"]["Path"] = args.input
e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

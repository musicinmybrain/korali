#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *

####### Parsing arguments
os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--run', help='Specifies the result folder to load the best policy from.', required=True)
parser.add_argument('--nn', help='Neural net width of two hidden layers.', required=False, type=int, default = 128)
parser.add_argument('--dis', help='Sampling Distribution.', required=False,type = str, default = 'Clipped Normal')

args = parser.parse_args()

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results

initEnvironment(e, args.env)
dis_dir = args.dis.replace(" ","_")
resultFolder = 'results/_result_vracer_' + args.env + '_' + dis_dir + '_'+ str(args.nn)+ '_' + str(args.run) +'/'
found = e.loadState(resultFolder  + '/latest');

if (found == False): 
 print('Error: could not find results in folder: ' + args.input)
 exit(-1)

### Enabling step information output

e["Problem"]["Custom Settings"]["Print Step Information"] = "Enabled"

### Setting file output configuration

e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = [ 0 ]
e["File Output"]["Path"] = resultFolder
e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

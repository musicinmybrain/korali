#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import json

def model(x, resultFolder, objective):
 
 SourceFolderName = "_config"
 DestinationFolderName = resultFolder + '/gen' + str(x["Current Generation"]).zfill(3) + '/sample' + str(x["Sample Id"]).zfill(6) 
 
 # Copy the 'model' folder into a temporary directory
 if os.path.exists( DestinationFolderName ):
  shutil.rmtree( DestinationFolderName)
 shutil.copytree( SourceFolderName, DestinationFolderName )

 CurrentDirectory = os.getcwd()

 # Move inside the temporary directory
 try:
  os.chdir( DestinationFolderName )
 except OSError as e:
  print("I/O error(" + str(e.errno) + "): " + e.strerror )
  print("The folder " + DestinationFolderName + " is missing")
  sys.exit(1)
 
 # Storing base parameter file
 configFile='par.py'
 with open(configFile, 'a') as f:
  f.write('arc_width = [ 1, %.10f, %.10f, %.10f, %.10f, %.10f, 1 ]\n' % ( x["Parameters"][0], x["Parameters"][1], x["Parameters"][2], x["Parameters"][3], x["Parameters"][4] ))
  f.write('arc_offset = [ 0, %.10f, %.10f, %.10f, %.10f, %.10f, 0 ]\n' % ( x["Parameters"][5], x["Parameters"][6], x["Parameters"][7], x["Parameters"][8], x["Parameters"][9] ))
  
 # Run Aphros for this sample
 sampleOutFile = open('sample.out', 'w')
 sampleErrFile = open('sample.err', 'w')
 subprocess.call(['bash', 'run.sh'], stdout=sampleOutFile, stderr=sampleErrFile)

 # Loading results from file
 resultFile = 'objectives'
 try:
  with open(resultFile) as f:
   resultContent = f.read()
 except IOError:
  print("[Korali] Error: Could not load result file: " + resultFile)
  exit(1)
 
 # Parsing output into JSON compatible format
 resultContent = resultContent.replace("'", '"').replace("True", "true").replace("False", "false")
 resultJs = json.loads(resultContent)

 # Declaring objective value as -inf, for the case of an invalid evaluation
 objectiveValue = float('-inf')

 # If sample is valid, evaluating result based on objective
 if (resultJs['valid'] == True):
  if (objective == 'minNumCoal'):
   objectiveValue = -float(resultJs['num_coal'])
  if (objective == 'maxNumCoal'):
   objectiveValue = float(resultJs['num_coal'])
  if (objective == 'maxNumCoal'):
   objectiveValue = float(resultJs['mean_velocity'])

 # Assigning objective function value
 x["F(x)"] = objectiveValue

 # Move back to the base directory
 os.chdir( CurrentDirectory )

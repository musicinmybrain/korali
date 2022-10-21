#!/usr/bin/env python3
from abf import *
import os

######## Defining Environment Storage

resDir = ''
maxReward = -100000
swimmers = Swimmers()

def setResultsDir(dir):
 if not os.path.exists(dir): os.makedirs(dir)
 global resDir
 resDir=dir

def env(s):

 # Getting sample id
 sampleId = s["Sample Id"]
 
 # Initializing environment
 swimmers.reset()
 
 s["State"] = swimmers.getState().tolist()
 
 cumulativeReward = 0
 done = False
 while not done:

  # Getting new action
  s.update()
  
  # Performing the action
  done = swimmers.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = swimmers.getReward()
   
  # Storing New State
  s["State"] = swimmers.getState().tolist()

  cumulativeReward = cumulativeReward + s["Reward"]
  
 # Setting finalization status
 if (swimmers.isSuccess()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
  
 # Saving the last/best trajectories for visualization
 global resDir
 global maxReward
 if (cumulativeReward > maxReward):
  maxReward = cumulativeReward
  swimmers.dumpTrajectoryToCsv(resDir + '/best.csv')
 swimmers.dumpTrajectoryToCsv(resDir + '/last.csv')
 
 # Saving current trajectory
 swimmers.dumpTrajectoryToCsv(resDir + '/trajectory' + str(sampleId).zfill(6) + '.csv')
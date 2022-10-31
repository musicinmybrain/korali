#!/usr/bin/env python3
from double_pendulum import *

######## Defining Environment Storage

upswing = DoublePendulum()
maxSteps = 1000

def env(s):

 # Initializing environment
 upswing.reset()
 s["State"] = upswing.getState().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = upswing.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = upswing.getReward()
   
  # Storing New State
  s["State"] = upswing.getState().tolist()

  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
  

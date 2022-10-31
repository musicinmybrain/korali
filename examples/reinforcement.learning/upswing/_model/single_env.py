#!/usr/bin/env python3
from single_pendulum import *

######## Defining Environment Storage

upswing = SinglePendulum()
maxSteps = 500

def env(s):

 # Initializing environment
 upswing.reset()
 s["State"] = upswing.getState().tolist()
 step = 0
 done = False

 maxs = np.ones(len(upswing.getState().tolist()))

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = upswing.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = upswing.getReward()
 
 # Storing New State
  state = upswing.getState().tolist()
  s["State"] = state

  maxs = np.maximum(maxs, state)
 
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
  

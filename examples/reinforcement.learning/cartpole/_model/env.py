#!/usr/bin/env python3
from cartpole import *
import pdb
import numpy as np
######## Defining Environment Storage
cart = CartPole()
maxSteps = 500

numActions = 1
theta = np.random.normal(loc=0., scale=0.01, size=(2,4))

def policy(state):
    global theta
    
    action = theta @ state
    return action


def env(sample):

 # Initializing environment and random seed
 sampleId = sample["Sample Id"]
 cart.reset(sampleId)
 sample["State"] = cart.getState().tolist()
 step = 0
 done = False

 if sample.contains("Mini Batch"):
     miniBatch = np.array(sample["Mini Batch"])

     stateSequenceBatch = np.array(sample["State Sequence Batch"])
     numBatch, effectiveMiniBatchSize, numStates, _ = stateSequenceBatch.shape

     policyParams = np.empty(shape = (numBatch, effectiveMiniBatchSize, 2*numActions), dtype=float)
     for b, batch in enumerate(stateSequenceBatch):
         for s, states in enumerate(batch):
                 policyParams[b,s,:] = policy(states[0])

     sample["Distribution Params"] = policyParams.tolist()

 while not done and step < maxSteps:
  
  # Calculate policy here and return action
  action = policy(cart.getState())
  sample["Action"] = action.tolist()
 
  # Getting new action
  sample.update()
 
  # Performing the action
  done = cart.advance(action)
  
  # Getting Reward
  sample["Reward"] = cart.getReward()
   
  # Storing New State
  sample["State"] = cart.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  sample["Termination"] = "Terminal"
 else:
  sample["Termination"] = "Truncated"

def multienv(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]
 if s["Mode"] == "Training":
    envId = sampleId % 3
 else:
    envId = 0
 cart.reset(sampleId * 1024 + launchId)
 s["State"] = cart.getState().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:
  
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  
  # Getting Reward
  
  s["Reward"] = cart.getReward()
  
  reward = cart.getReward()
  if (envId == 0):
    s["Reward"] = cart.getReward()
  elif (envId == 1):
    s["Reward"] = cart.getReward() - 1
  else:
    s["Reward"] = cart.getReward() * 0.1
   
  # Storing New State
  s["State"] = cart.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

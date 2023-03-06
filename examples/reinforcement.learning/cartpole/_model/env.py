#!/usr/bin/env python3
from cartpole import *
import pdb
import numpy as np
######## Defining Environment Storage
cart = CartPole()
maxSteps = 500

numActions = 1
theta = np.random.normal(loc=0., scale=0.01, size=(2,4))

# Primitive policy
def policy(state):
    global theta
    muSigma = theta @ state
    muSigma[numActions:] = np.exp(muSigma[numActions:])
    action = np.random.normal(loc=muSigma[:numActions], scale=muSigma[numActions:])
    return action, muSigma.flatten()


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
                 _, policyParams[b,s,:] = policy(states[0])

     print("x")
     exit()
     sample["Distribution Parameters"] = policyParams.tolist()

 while not done and step < maxSteps:
  
  # Calculate policy here and return action
  action, distParams = policy(cart.getState())
  sample["Action"] = action.tolist()
  sample["Distribution Parameters"] = distParams.tolist()
  print(step)
  print(sample["Distribution Parameters"])
 
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
 print(sample["Termination"])

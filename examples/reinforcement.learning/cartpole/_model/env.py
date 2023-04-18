#!/usr/bin/env python3
from cartpole import *
import pdb
import numpy as np
######## Defining Environment Storage
cart = CartPole()
maxSteps = 500

numActions = 1
numAgents = 2
numPolicyParams = 2
learningRate = 0.0001
theta = np.random.normal(loc=0., scale=0.0001, size=(2,4))

# Primitive policy
def policy(state):
    global theta
    # linear
    muSigma = theta @ state
    # sigmoid
    muSigma[numActions:] = np.exp(muSigma[numActions:]) / (1 + np.exp(muSigma[numActions:]))
    action = np.random.normal(loc=muSigma[:numActions], scale=muSigma[numActions:])
    return action, muSigma.flatten()

def env(sample):

 # If sample contains gradient, do the gradient update
 if sample.contains("Gradients"):
     #print("TODO: Received gradient, update external policy!")
     global theta
     mb = np.array(sample["Gradients"]["Mini Batch"])
     #print("MB shape")
     #print(mb.shape)
     nupdate, nmb, _ = mb.shape
     expstates = np.array(sample["Gradients"]["State Sequence Batch"])[:,:,0,:]
     #print("SSB shape")
     #print(expstates[0,0,:])
     #print(expstates[0,1,:])
     #print(expstates[0,2,:])
     #print(expstates[0,3,:])
     #print(expstates.shape)
     gradients = np.array(sample["Gradients"]["Gradients"])
     #print(gradients[0,0,:])
     #print(gradients[0,1,:])
     #print(gradients[0,2,:])
     #print(gradients[0,3,:])
     #print(theta)
     #print("TH shape")
     #print(theta.shape)
     #print(gradients)
     #print("GRAD shape")
     #print(gradients.shape)

     thetaOld = theta
     for i in range(nupdate):
        for b in range(0, nmb):
            # update mu params
            theta[0,:] += learningRate/nmb * expstates[i,b,:] * gradients[i,b,0]
            # update sig params
            tmp = np.exp(expstates[i,b,:] @ thetaOld[1,:]) 
            dsig = tmp * (1. - tmp)
            theta[1,:] += learningRate/nmb * dsig * expstates[i,b,:] * gradients[i,b,1]

 # If sample contains mini-batch, evaluate the state sequence and return distribution params
 if sample.contains("Mini Batch"):
     miniBatch = np.array(sample["Mini Batch"])
     #print("Mini Batch received, evaluating state sequences..")
     #print(miniBatch.shape)

     stateSequenceBatch = np.array(sample["State Sequence Batch"])
     #print(stateSequenceBatch.shape)
     numBatch, effectiveMiniBatchSize, numStates, _ = stateSequenceBatch.shape

     policyParams = np.empty(shape = (numBatch, effectiveMiniBatchSize, 2*numActions), dtype=float)
     for b, batch in enumerate(stateSequenceBatch):
         for s, states in enumerate(batch):
                 _, policyParams[b,s,:] = policy(states[0])

     #print("PP shape")
     #print(policyParams.shape)
     sample["Distribution Parameters"] = policyParams.tolist()
     sample["Termination"] = "Terminal"
     return
 
 # Initializing environment and random seed
 sampleId = sample["Sample Id"]
 cart.reset(sampleId)
 sample["State"] = [cart.getState().tolist(), cart.getState().tolist()]
 step = 0
 done = False

 # Else run episode
 while not done and step < maxSteps:
  # Calculate policy here and return action
  action, distParams = policy(cart.getState())
  sample["Action"] = [action.tolist(), action.tolist()]
  sample["Distribution Parameters"] = [distParams.tolist(), distParams.tolist()]
 
  # Getting new action
  sample.update()
 
  # Performing the action
  done = cart.advance(action)
  
  # Getting Reward
  sample["Reward"] = [cart.getReward(), cart.getReward()]
   
  # Storing New State
  sample["State"] = [cart.getState().tolist(), cart.getState().tolist()]
  
  # Advancing step counter
  step = step + 1

 _, distParams = policy(cart.getState())
 sample["Distribution Parameters"] = distParams.tolist()
 
 # Setting finalization status
 if (cart.isOver()):
  sample["Termination"] = "Terminal"
 else:
  sample["Termination"] = "Truncated"
 
 return

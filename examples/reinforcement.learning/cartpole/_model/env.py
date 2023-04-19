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
theta = np.random.normal(loc=0., scale=0.0001, size=(4,5))

# Primitive policy
def policy(state):
    global theta
    tmp = theta[:,:4] @ state + theta[:,-1]
    muSigma = np.zeros(numPolicyParams)
    muSigma[:numActions] = tmp[0] + np.exp(tmp[1]) + np.exp(-tmp[2])
    # sigmoid
    muSigma[numActions:] = np.exp(tmp[3]) / (1 + np.exp(tmp[3]))
    action = np.random.normal(loc=muSigma[:numActions], scale=muSigma[numActions:])
    return action, muSigma.flatten()

def env(sample):

 # If sample contains gradient, do the gradient update
 if sample.contains("Gradients"):
     #print("TODO: Received gradient, update external policy!")
     global theta
     mb = np.array(sample["Gradients"]["Mini Batch"])
     nupdate, nmb, _ = mb.shape
     expstates = np.array(sample["Gradients"]["State Sequence Batch"])[:,:,0,:]
     gradients = np.array(sample["Gradients"]["Gradients"])

     thetaOld = theta
     for i in range(nupdate):
        for b in range(0, nmb):
            tmp = thetaOld[:,:4] @ expstates[i,b,:] + thetaOld[:,-1]
            tmp1 = np.exp(tmp[1])
            tmp2 = np.exp(-tmp[2])
            
            # update mu params
            theta[0,:4] += 1./(nmb*numAgents) * expstates[i,b,:] * gradients[i,b,0]
            theta[0,-1] += 1./(nmb*numAgents) * gradients[i,b,0]

            theta[1,:4] += 1./(nmb*numAgents) * tmp1 * expstates[i,b,:] * gradients[i,b,0]
            theta[1,-1] += 1./(nmb*numAgents) * tmp1 * gradients[i,b,0]
            
            theta[2,:4] += 1./(nmb*numAgents) * tmp2 * -1 * expstates[i,b,:] * gradients[i,b,0]
            theta[2,-1] += 1./(nmb*numAgents) * tmp2 * -1 * gradients[i,b,0]

            # update sig params
            tmp3 = np.exp(tmp[3]) / (1+np.exp(tmp[3]))
            dsig = tmp3 * (1. - tmp3)
            theta[3,:4] += 1./(nmb*numAgents) * dsig * expstates[i,b,:] * gradients[i,b,1]
            theta[3,-1] += 1./(nmb*numAgents) * dsig * gradients[i,b,1]

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

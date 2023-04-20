#!/usr/bin/env python3
from cartpole import *
import pdb
import numpy as np

######## Setup Policy
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
stateDim = 4
actionDim = 1
hiddenLayers = [32,32]
activationFunction = 'tanh'

inputs = tf.keras.Input(shape=(stateDim,), dtype='float32')
for i, size in enumerate(hiddenLayers):
    if i == 0:
        x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation=activationFunction, dtype='float32')(inputs)
    else:
        x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation=activationFunction, dtype='float32')(x)

scaledGlorot = lambda shape, dtype : 0.001*tf.keras.initializers.GlorotNormal()(shape)

mean  = tf.keras.layers.Dense(actionDim, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
sigma = tf.keras.layers.Dense(actionDim, kernel_initializer=scaledGlorot, activation = "softplus", dtype='float32')(x)

outputs = tf.keras.layers.Concatenate()([mean, sigma])
policyNetwork = tf.keras.Model(inputs=inputs, outputs=outputs, name='PolicyNetwork')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

######## Defining Environment Storage
cart = CartPole()
maxSteps = 500

numActions = 1
numAgents = 1
numPolicyParams = 2

# Primitive policy
def policy(state):
    global policyNetwork
    tfstate = tf.convert_to_tensor([state])
    meanSigma = policyNetwork(tfstate)
    meanSigma = meanSigma.numpy().tolist()
    action = np.random.normal(loc=meanSigma[0][:numActions], scale=meanSigma[0][numActions:])
    return action.tolist(), meanSigma[0]

def gradpolicy(state):
    global policyNetwork
    tfstate = tf.convert_to_tensor([state])
    with tf.GradientTape() as tape:
      tape.watch(policyNetwork.trainable_weights)
      meanSigma = policyNetwork(tfstate)
    grad = tape.gradient(meanSigma,policyNetwork.trainable_weights)
    return grad


def env(sample):

 # If sample contains gradient, do the gradient update
 if sample.contains("Gradients"):
     print("Received gradient, update external policy")
     global policyNetwork

     mb = np.array(sample["Gradients"]["Mini Batch"])
     nupdate, nmb, _ = mb.shape
     expstates = np.array(sample["Gradients"]["State Sequence Batch"])[:,:,0,:]
     gradients = np.array(sample["Gradients"]["Gradients"])
  
     print(policyNetwork.trainable_weights[7])

     for i in range(nupdate):
        for b in range(0, nmb):
            gradw = gradpolicy(expstates[i,b,:])
            for gw in range(len(gradw)-4):
                for g in range(len(gradients[i,b,:])):
            	    policyNetwork.trainable_weights[gw].assign(policyNetwork.trainable_weights[gw] + 1./(nmb*numAgents) * gradw[gw] * gradients[i,b,g])
            #print(gw)
            #print(policyNetwork.trainable_weights[4])
            #print(1./(nmb*numAgents) * gradw[4] * gradients[i,b,0])
            #print(policyNetwork.trainable_weights[4] + 10./(nmb*numAgents) * gradw[4] * gradients[i,b,0])
            #exit()
            policyNetwork.trainable_weights[4].assign(policyNetwork.trainable_weights[4] + 1./(nmb*numAgents) * gradw[4] * gradients[i,b,0])
            policyNetwork.trainable_weights[5].assign(policyNetwork.trainable_weights[5] + 1./(nmb*numAgents) * gradw[5] * gradients[i,b,0])
            policyNetwork.trainable_weights[6].assign(policyNetwork.trainable_weights[6] + 1./(nmb*numAgents) * gradw[6] * gradients[i,b,1])
            policyNetwork.trainable_weights[7].assign(policyNetwork.trainable_weights[7] + 1./(nmb*numAgents) * gradw[7] * gradients[i,b,1])

 # If sample contains mini-batch, evaluate the state sequence and return distribution params
 if sample.contains("Mini Batch"):
     print("Received Mini Batch, evaluate state sequence batch!")
     miniBatch = np.array(sample["Mini Batch"])

     stateSequenceBatch = np.array(sample["State Sequence Batch"])
     #print(stateSequenceBatch.shape)
     numBatch, effectiveMiniBatchSize, numStates, _ = stateSequenceBatch.shape

     policyParams = np.empty(shape = (numBatch, effectiveMiniBatchSize, 2*numActions), dtype=float)
     for b, batch in enumerate(stateSequenceBatch):
         for s, states in enumerate(batch):
             _, policyParams[b,s,:] = policy(states[0])

     sample["Distribution Parameters"] = policyParams.tolist()
     sample["Termination"] = "Terminal"
     return
 
 # Initializing environment and random seed
 sampleId = sample["Sample Id"]
 cart.reset(sampleId)
 sample["State"] = cart.getState().tolist()
 step = 0
 done = False

 # Else run episode
 while not done and step < maxSteps:
  # Calculate policy here and return action
  action, distParams = policy(cart.getState())
  sample["Action"] = action
  sample["Distribution Parameters"] = distParams
 
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

 _, distParams = policy(cart.getState())
 sample["Distribution Parameters"] = distParams
 # Setting finalization status
 if (cart.isOver()):
  sample["Termination"] = "Terminal"
 else:
  sample["Termination"] = "Truncated"
 
 return

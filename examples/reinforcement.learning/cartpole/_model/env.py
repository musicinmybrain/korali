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

scaledGlorot = lambda shape, dtype : 0.0001*tf.keras.initializers.GlorotNormal()(shape)

mean  = tf.keras.layers.Dense(actionDim, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
sigma = tf.keras.layers.Dense(actionDim, kernel_initializer=scaledGlorot, activation = "softplus", dtype='float32')(x)

outputs = tf.keras.layers.Concatenate()([mean, sigma])
policyNetwork = tf.keras.Model(inputs=inputs, outputs=outputs, name='PolicyNetwork')
policyNetwork.compile(optimizer='adam')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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
    action = np.clip(np.random.normal(loc=meanSigma[0][:numActions], scale=meanSigma[0][numActions:]), a_min=-10, a_max=10)
    return action.tolist(), meanSigma[0]

def gradpolicy(sample):
    global optimizer
    global policyNetwork

    states = np.array(sample["Gradients"]["State Sequence Batch"])[:,0,:]
    lossGradients = np.array(sample["Gradients"]["Gradients"])

    policyNetworkTmp = tf.keras.Model(inputs=inputs, outputs=outputs, name='PolicyNetwork')
    for wtmp, w, in zip(policyNetworkTmp.trainable_weights, policyNetwork.trainable_weights):
        wtmp.assign(w)
    
    # Tmp storage for weight updates
    trainableWeights = policyNetwork.get_weights().copy()

    mb1, _ = states.shape 
    mb2, _ = lossGradients.shape 
    assert(mb1 == mb2)
    for b in range(mb1):
        tfstate = tf.convert_to_tensor([states[b,:]])
        with tf.GradientTape() as tape:
            tape.watch(policyNetwork.trainable_weights)
            meanSigma = policyNetwork(tfstate)

        gradw = tape.gradient(meanSigma,policyNetwork.trainable_weights)
        nlay = len(gradw)

        for gw in range(nlay-4):
            trainableWeights[gw] += 1./(numAgents) * gradw[gw] * lossGradients[b,0]
            trainableWeights[gw] += 1./(numAgents) * gradw[gw] * lossGradients[b,1]
        trainableWeights[4] += 1./(numAgents) * gradw[4] * lossGradients[b,0]
        trainableWeights[5] += 1./(numAgents) * gradw[5] * lossGradients[b,0]
        trainableWeights[6] += 1./(numAgents) * gradw[6] * lossGradients[b,1]
        trainableWeights[7] += 1./(numAgents) * gradw[7] * lossGradients[b,1]
    
    # Copy back variables
    for wtmp, w, in zip(trainableWeights, policyNetwork.trainable_weights):
        w.assign(wtmp)

def env(sample):

 # If sample contains gradient, do the gradient update
 if sample.contains("Gradients"):
     #print("Received gradient, update external policy")
     gradpolicy(sample)

 # If sample contains mini-batch, evaluate the state sequence and return distribution params
 if sample.contains("Mini Batch"):
     #print("Received Mini Batch, evaluate state sequence batch!")
     miniBatch = np.array(sample["Mini Batch"])
     stateSequenceBatch = np.array(sample["State Sequence Batch"])
     effectiveMiniBatchSize, numStates, _ = stateSequenceBatch.shape

     policyParams = np.empty(shape = (effectiveMiniBatchSize, 2*numActions), dtype=float)
     for s, states in enumerate(stateSequenceBatch):
         _, policyParams[s,:] = policy(states[0])
             

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

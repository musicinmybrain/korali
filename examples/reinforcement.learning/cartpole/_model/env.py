#!/usr/bin/env python3
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)

from cartpole import *
import numpy as np
import time

######## Define Policy
numActions = 1
numAgents = 1
stateDim = 4

hiddenLayers = [32,32]
inputs = tf.keras.Input(shape=(stateDim,), dtype='float32')
for i, size in enumerate(hiddenLayers):
    if i == 0:
        x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation='tanh', dtype='float32')(inputs)
    else:
        x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation='tanh', dtype='float32')(x)

scaledGlorot = lambda shape, dtype : 0.001*tf.keras.initializers.GlorotNormal()(shape)
value = tf.keras.layers.Dense(1, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
mean  = tf.keras.layers.Dense(numActions, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
sigma = tf.keras.layers.Dense(numActions, kernel_initializer=scaledGlorot, activation = "softplus", dtype='float32')(x)
outputs = tf.keras.layers.Concatenate()([value, mean, sigma])
policyNetwork = tf.keras.Model(inputs=inputs, outputs=outputs, name='PolicyNetwork')
policyNetwork.compile(optimizer='adam')

# Primitive policy
def policy(state):
    global policyNetwork
    tfstate = tf.convert_to_tensor([state])
    valueMeanSigma = policyNetwork(tfstate).numpy().tolist()
    action = np.clip(np.random.normal(loc=valueMeanSigma[0][1:numActions+1], scale=valueMeanSigma[0][1+numActions:]), a_min=-10, a_max=10)
    return action.tolist(), valueMeanSigma[0][0], valueMeanSigma[0][1:]

# Very primitive policy update with SGD
def policyUpdate(sample):
    global policyNetwork

    states = np.array(sample["Gradients"]["State Sequence Batch"])[:,0,:] # The 'sequence' is of size 1
    lossGradients = np.array(sample["Gradients"]["Gradients"])

    mb1, _ = states.shape 
    mb2, _ = lossGradients.shape 
    assert(mb1 == mb2)

    policyNetworkTmp = tf.keras.Model(inputs=inputs, outputs=outputs, name='PolicyNetwork')
    for wtmp, w, in zip(policyNetworkTmp.trainable_weights, policyNetwork.trainable_weights):
        wtmp.assign(w)
    
    # Tmp storage for weight updates
    trainableWeights = policyNetwork.get_weights().copy()

    print(lossGradients)
    for b in range(mb1):
        # here we forward again, this can be optimized
        tfstate = tf.convert_to_tensor([states[b,:]])
        with tf.GradientTape() as tape:
            tape.watch(policyNetwork.trainable_weights)
            valueMeanSigma = policyNetwork(tfstate)

        gradw = tape.gradient(valueMeanSigma,policyNetwork.trainable_weights)
        nlay = len(gradw)

        for gw in range(nlay-6):
            trainableWeights[gw] += 1./mb1 * gradw[gw] * lossGradients[b,0]
            trainableWeights[gw] += 1./mb1 * gradw[gw] * lossGradients[b,1]
            trainableWeights[gw] += 1./mb1 * gradw[gw] * lossGradients[b,2]

        # weight updates last layer for mean output
        trainableWeights[4] += 1./mb1 * gradw[4] * lossGradients[b,0]
        trainableWeights[5] += 1./mb1 * gradw[5] * lossGradients[b,0]
        # weight updates last layer for mean output
        trainableWeights[6] += 1./mb1 * gradw[6] * lossGradients[b,1]
        trainableWeights[7] += 1./mb1 * gradw[7] * lossGradients[b,1]
        # weight updates last layer for sigma output
        trainableWeights[8] += 1./mb1 * gradw[8] * lossGradients[b,2]
        trainableWeights[9] += 1./mb1 * gradw[9] * lossGradients[b,2]
    
    # Copy back variables
    for w,wtmp  in zip(policyNetwork.trainable_weights, trainableWeights):
        w.assign(wtmp)


######## Profiling
tstart = time.time()
tpolupdate = 0.
tmbeval = 0.
tenv  = 0.


######## Defining Environment Storage
cart = CartPole()
maxSteps = 500

def env(sample):
 global tstart
 global tpolupdate
 global tmbeval
 global tenv

 time0  = time.time()
 # If sample contains gradient, update the policy
 if sample.contains("Gradients"):
     policyUpdate(sample)

 time1  = time.time()
 tpolupdate += (time1-time0)
 # If sample contains mini-batch, evaluate the state sequence and return distribution params
 if sample.contains("Mini Batch"):
     miniBatch = np.array(sample["Mini Batch"])
     stateSequenceBatch = np.array(sample["State Sequence Batch"])
     effectiveMiniBatchSize, numStates, _ = stateSequenceBatch.shape

     values = np.empty(shape = (effectiveMiniBatchSize), dtype=float)
     policyParams = np.empty(shape = (effectiveMiniBatchSize, 2*numActions), dtype=float)
     for s, states in enumerate(stateSequenceBatch):
         _, values[s], policyParams[s,:] = policy(states[0])
     
     sample["State Value"] = values.tolist()
     sample["Distribution Parameters"] = policyParams.tolist()
     sample["Termination"] = "Terminal"
     # Important: Exit after mini batch evaluation, rest will ignored during policy update
     return

 time2  = time.time()
 tmbeval += (time2-time1)
 # Initializing environment and random seed
 sampleId = sample["Sample Id"]
 cart.reset(sampleId)
 sample["State"] = cart.getState().tolist()
 step = 0
 done = False

 # Else run episode
 while not done and step < maxSteps:
  # Calculate policy here and return action
  action, value, distParams = policy(cart.getState())
  sample["Action"] = action
  sample["State Value"] = value
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

 _, value, distParams = policy(cart.getState())
 sample["State Value"] = value
 sample["Distribution Parameters"] = distParams
 # Setting finalization status
 if (cart.isOver()):
  sample["Termination"] = "Terminal"
 else:
  sample["Termination"] = "Truncated"

 time3 = time.time()
 tenv += (time3-time2)
 ttotal = time3 - tstart
 
 print(f"pct pol update: \t{100*tpolupdate/ttotal:.1f}")
 print(f"pct mb eval: \t\t{100*tmbeval/ttotal:.1f}")
 print(f"pct env: \t\t{100*tenv/ttotal:.1f}")
 print(f"pct py: \t\t{100*(tpolupdate+tmbeval+tenv)/ttotal:.1f}")
 print(f"ttotal: \t\t{ttotal:.1f}")
 return

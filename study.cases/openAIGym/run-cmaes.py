#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

import sys
import math
import numpy as np

sys.path.append('_model')
import gym
#import pyBulletEnvironments
from HumanoidWrapper import HumanoidWrapper
from AntWrapper import AntWrapper

# Define Parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--population', help='Population Size.', type=int, required=True)
parser.add_argument('--diag', help='Diagonal Covariance Matrix adaption.', action='store_true')
parser.add_argument('--mirror', help='Mirror Sampling.', action='store_true')
parser.add_argument('--weight', help='Mu Weightning.', default="Logarithmic", type=str)

previousStateList = []
currentStateList = []

def agent(s, env, mirror, populationSize):
 
 stateVariableCount = env.observation_space.shape[0]
 actionVariableCount = env.action_space.shape[0]
 
 # Get policy
 X = np.array(s["Parameters"])
 X = X.reshape((actionVariableCount, stateVariableCount))
 
 # Data Whitening
 global previousStateList
 global currentStateList
 if len(previousStateList) == 0:
     mu = np.zeros(stateVariableCount)
     Sigma = np.ones(stateVariableCount)
     SigmaInv = 1./Sigma
 else:
     stateArr = np.array([item for sublist in previousStateList for item in sublist])
     mu = np.mean(stateArr, axis=0)
     Sigma = np.std(stateArr, axis=0)
     SigmaInv = 1./Sigma

 # Init rollout
 state = env.reset()
 step = 0
 done = False
 
 states = [np.array(state)]

 # Storage for cumulative reward
 cumulativeReward = 0.0
 overSteps = 0
 while not done and step < 1000:
  # Performing the action
  stateHat = state - mu
  M = np.multiply(X,SigmaInv)
  action = np.dot(M, stateHat)
  state, reward, done, _ = env.step(action)
  states.append(np.array(state))
  
  # Update cumulative reward
  cumulativeReward = cumulativeReward + reward 
  
  # Advancing step counter
  step = step + 1
 
 currentStateList.append(states)
 # Reset state lists
 if len(currentStateList) == populationSize:
  previousStateList = currentStateList
  currentStateList = []

 s["F(x)"] = cumulativeReward

if __name__ == '__main__':
  
  args = parser.parse_args()

  envName = args.env
  populationSize = args.population
  diag = args.diag
  muweight = args.weight
  mirror = args.mirror

  resultFolder = '_result_cmaes_{}_{}_diag{}_mu-{}_mirror{}/'.format(envName, populationSize, diag, muweight, mirror)
 
  # Environment specifics
  env = gym.make(envName)
  if (envName == 'Humanoid-v2'):
    env = HumanoidWrapper(env)
  if (envName == 'HumanoidStandup-v2'):
    env = HumanoidWrapper(env)
  if (envName == 'Ant-v2'):
    env = AntWrapper(env)
   
  stateVariableCount = env.observation_space.shape[0]
  actionVariableCount = env.action_space.shape[0]
   
  stateVariablesIndexes = range(stateVariableCount)
  if (envName == 'Ant-v2'):
    stateVariableCount = 27
    stateVariablesIndexes = range(stateVariableCount) 


  # Starting Korali's Engine
  import korali
  k = korali.Engine()
  e = korali.Experiment()

  #e.loadState(resultFolder + '/latest');

  e["Random Seed"] = 0xC0FEE
  e["Problem"]["Type"] = "Optimization"
  e["Problem"]["Objective Function"] = lambda s : agent(s, env, mirror, populationSize)

  dim = stateVariableCount * actionVariableCount

  # Defining the problem's variables.
  for i in range(dim):
      e["Variables"][i]["Name"] = "X" + str(i)
      e["Variables"][i]["Lower Bound"] = -5.0
      e["Variables"][i]["Upper Bound"] = +5.0
      e["Variables"][i]["Initial Standard Deviation"] = 2e-2

  # Configuring CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/CMAES"
  e["Solver"]["Population Size"] = populationSize
  e["Solver"]["Mu Value"] = populationSize
  e["Solver"]["Mu Type"] = muweight
  e["Solver"]["Diagonal Covariance"] = diag
  e["Solver"]["Mirrored Sampling"] = mirror
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-32
  e["Solver"]["Termination Criteria"]["Max Generations"] = 1000

  # Configuring results path
  e["File Output"]["Enabled"] = True
  e["File Output"]["Path"] = resultFolder
  e["File Output"]["Frequency"] = 10

  # Running Korali
  k.run(e)

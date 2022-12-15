#!/usr/bin/env python3
import sys
import gym
import numpy as np
from HumanoidWrapper import HumanoidWrapper
from AntWrapper import AntWrapper
from packaging.version import parse as parse_version

if (gym.__version__ != "0.26.2"):
    print(f"[agent] Gym version 0.26.2 expected, but is {gym.__version__}")
    print("[agent] Exit..")
    sys.exit()

def initEnvironment(e, envName, excludePositions, moviePath = ''):
 
 # Creating environment 
 
 if (envName == 'Reacher-v4'):
  env = gym.make(envName)
 elif (envName == 'Ant-v4'):
  env = gym.make(envName)
 else:
  env = gym.make(envName, exclude_current_positions_from_observation=excludePositions)

 oldEnv = parse_version(gym.__version__) < parse_version('0.26.0')

 if (envName == 'Humanoid-v4'):
  env = HumanoidWrapper(env)
 
 if (envName == 'HumanoidStandup-v4'):
  env = HumanoidWrapper(env)
 
 if (envName == 'Ant-v4'):
  env = AntWrapper(env)
 
 # Re-wrapping if saving a movie
 if (moviePath != ''):
  env = gym.wrappers.Monitor(env, moviePath, force=True)
 
 ### Defining problem configuration for openAI Gym environments
 e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
 e["Problem"]["Environment Function"] = lambda x : agent(x, env, excludePositions)
 e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
 
 # Getting environment variable counts

 stateVariableCount = env.observation_space.shape[0] if excludePositions else env.observation_space.shape[0] - 1
 actionVariableCount = env.action_space.shape[0]
 
 # Generating state variable index list
 stateVariablesIndexes = range(stateVariableCount)
 
 # Defining State Variables
 for i in stateVariablesIndexes:
  e["Variables"][i]["Name"] = "State Variable " + str(i)
  e["Variables"][i]["Type"] = "State"
  e["Variables"][i]["Lower Bound"] = float(env.observation_space.low[i])
  e["Variables"][i]["Upper Bound"] = float(env.observation_space.high[i])
 
 # Defining Action Variables
 
 for i in range(actionVariableCount):
  e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
  e["Variables"][stateVariableCount + i]["Type"] = "Action"
  e["Variables"][stateVariableCount + i]["Lower Bound"] = float(env.action_space.low[i])
  e["Variables"][stateVariableCount + i]["Upper Bound"] = float(env.action_space.high[i])
  e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = 0.4472

def agent(s, env, excludePositions):

 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False
 
 state = env.reset()[0]
 s["State"] = state.tolist() if excludePositions else state.tolist()[1:]
 s["Features"] = state.tolist()

 step = 0
 done = False
 
 # Storage for cumulative reward
 cumulativeReward = 0.0
 
 overSteps = 0
 
 while not done and step < 1000:
  # Getting new action
  s.update()
  
  # Printing step information
  #if (printStep):  print('[Korali] Frame ' + str(step), end = '')
  
  # Performing the action
  action = s["Action"]
 
  s["Features"] = state.tolist()
  state, reward, done, = env.step(action)[:3]
 
  # Getting Reward
  s["Reward"] = reward
  
  # Printing step information
  #if (printStep):  print(' - State: ' + str(state) + ' - Action: ' + str(action))
  cumulativeReward = cumulativeReward + reward 
  #if (printStep):  print(' - Cumulative Reward: ' + str(cumulativeReward))
  
  # Storing New State
  s["State"] = state.tolist() if excludePositions else state.tolist()[1:]
  
  # Advancing step counter
  step = step + 1

 if (printStep):  print(' - Cumulative Reward: ' + str(cumulativeReward))
 # Setting termination status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

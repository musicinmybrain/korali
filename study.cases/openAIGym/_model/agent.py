#!/usr/bin/env python3

import gym
import json
import os.path
from HumanoidWrapper import HumanoidWrapper
from AntWrapper import AntWrapper

from packaging.version import parse as parse_version
oldEnv = parse_version(gym.__version__) < parse_version('0.26.0')


def initEnvironment(e, envName, excludePosition=True, moviePath = ''):

 # Creating environment 
 
 if envName == "Reacher-v4":
    env = gym.make(envName)
    excludePosition = True
 else:
    #env = gym.make(envName, reset_noise_scale=0.3, exclude_current_positions_from_observation=excludePosition)
    #env = gym.make(envName, exclude_current_positions_from_observation=excludePosition)
    env = gym.make(envName)
 
 # Handling special cases

 if (envName == 'Humanoid-v2'):
  env = HumanoidWrapper(env)

 if (envName == 'HumanoidStandup-v2'):
  env = HumanoidWrapper(env)

 if (envName == 'Ant-v2'):
  env = AntWrapper(env)

 # Re-wrapping if saving a movie
 if (moviePath != ''):
  env = gym.wrappers.Monitor(env, moviePath, force=True)

 ### Defining problem configuration for openAI Gym environments
 e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
 e["Problem"]["Environment Function"] = lambda s : environment(s, env, excludePosition)
 e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
 e["Problem"]["Custom Settings"]["Save State"] = "False"
 
 # Getting environment variable counts
 if excludePosition == True:
   stateVariableCount = env.observation_space.shape[0]
 else:
   stateVariableCount = env.observation_space.shape[0] - 1

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
  e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = 0.2

def environment(s, env, excludePosition):
 
 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False
 
 if (s["Custom Settings"]["Save State"] == "True"):
   fname = s["Custom Settings"]["File Name"]
   saveState = True
   if os.path.isfile(fname):
     with open(fname, 'r') as infile:
       obsjson = json.load(infile)
       obsstates = obsjson["States"]
       obsactions = obsjson["Actions"]
       obsrewards = obsjson["Rewards"]
   else:
       obsjson = {}
       obsstates = []
       obsactions = []
       obsrewards = []
 
 else:
   saveState = False
   obsjson = {}
   obsstates = []
   obsactions = []
  
 if excludePosition == False:
    s["State"] = env.reset()[0].tolist()[1:]
 else:
    s["State"] = env.reset()[0].tolist()
 
 step = 0
 done = False
 
 # Storage for cumulative reward
 cumulativeReward = 0.0
 overSteps = 0
 
 states = []
 actions = []
 
 while not done and step < 1000:
  
  # Getting new action
  s.update()
  
  # Printing step information
  if (printStep):  print(f'[Korali] Frame {step}')
  
  # Performing the action
  action = s["Action"]
  state, reward, done = env.step(action)[:3]
  
  # Getting Reward
  s["Reward"] = reward
  
  # Printing step information
  #if (printStep):  print(' - State: ' + str(state) + ' - Action: ' + str(action))
  cumulativeReward = cumulativeReward + reward 
  if (printStep): print(f' - Reward: {reward}')
  if (saveState): 
      states.append(state.tolist())
      actions.append(action)
  
  # Storing New State
  if excludePosition == False:
    s["State"] = env.reset()[0].tolist()[1:]
  else:
    s["State"] = env.reset()[0].tolist()
  
  # Advancing step counter
  step = step + 1
 
 if (printStep):  print(f' - Cumulative Reward: {cumulativeReward}')
 if (saveState): 
   obsstates.append(states)
   obsactions.append(actions)
   obsrewards.append(cumulativeReward)
   obsjson["States"] = obsstates
   obsjson["Actions"] = obsactions
   obsjson["Rewards"] = obsrewards
   with open(fname, 'w') as f:
     json.dump(obsjson, f)
 
 # Setting termination status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

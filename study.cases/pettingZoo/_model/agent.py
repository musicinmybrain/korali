#!/usr/bin/env python3
#modification: multiwalker_v7 to multiwalker_v9; Add Simple Adversary from MPE environment
import math
import pdb
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def initEnvironment(e, envName, multPolicies):

 # Creating environment
 if (envName ==  'Waterworld'):
    from pettingzoo.sisl import waterworld_v3
    env = waterworld_v3.env()
    stateVariableCount = 242
    actionVariableCount = 2
    ac_upper = 0.01
    ac_low = -0.01
    numIndividuals = 5
 elif (envName == 'Multiwalker'):
    from pettingzoo.sisl import multiwalker_v9
    env = multiwalker_v9.env()
    stateVariableCount = 31
    actionVariableCount = 4
    ac_upper = 1
    ac_low = -1
    numIndividuals = 3
 elif (envName ==  'Pursuit'):
   from pettingzoo.sisl import pursuit_v4
   env = pursuit_v4.env()
   stateVariableCount = 147
   actionVariableCount = 1
   numIndividuals = 8
   possibleActions = [ [0], [1], [2], [3], [4] ]
 elif (envName == 'Simpletag'):
   from pettingzoo.mpe import simple_tag_v2
   env = simple_tag_v2.env(num_good=1, num_adversaries=1,continuous_actions=True)
   stateVariableCount = 62
   actionVariableCount = 5
   ac_upper = 1
   ac_low = 0
   numIndividuals = 2
 else:
   print("Environment '{}' not recognized! Exit..".format(envName))
   sys.exit()


 ## Defining State Variables

 ## what should we do with two different state variable counts in one environment
 for i in range(stateVariableCount):
   e["Variables"][i]["Name"] = "State Variable " + str(i)
   e["Variables"][i]["Type"] = "State"

 ## Defining Action Variables
 for i in range(actionVariableCount):
   e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
   e["Variables"][stateVariableCount + i]["Type"] = "Action"

# add the MPE to the continuous environment sets
 if (envName == 'Waterworld') or (envName == 'Multiwalker') or (envName == 'Simpletag'):
   ### Defining problem configuration for continuous environments
   e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
   e["Problem"]["Environment Function"] = lambda x : agent(x, env)
   e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
   e["Problem"]["Agents Per Environment"] = numIndividuals
   if (multPolicies == 1) :
      e["Problem"]["Policies Per Environment"] = numIndividuals

   # Defining Action Variables
   for i in range(actionVariableCount):
      e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
      e["Variables"][stateVariableCount + i]["Type"] = "Action"
      e["Variables"][stateVariableCount + i]["Lower Bound"] = float(ac_low)
      e["Variables"][stateVariableCount + i]["Upper Bound"] = float(ac_upper)
      e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = math.sqrt(0.2) * (ac_upper - ac_low)

 elif (envName ==  'Pursuit'):
   ### Defining problem configuration for discrete environments
   e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
   e["Problem"]["Environment Function"] = lambda x : agent(x, env)
   e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
   e["Problem"]["Possible Actions"] = possibleActions
   e["Problem"]["Agents Per Environment"] = numIndividuals
   if (multPolicies == 1) :
      e["Problem"]["Policies Per Environment"] = numIndividuals

 return numIndividuals

def agent(s, env):

 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False

 env.reset()

 states = []

# add MPE here
 if (env.env.env.metadata['name'] == 'pursuit_v4'):
    for ag in env.agents:
       state = env.observe(ag)
       state = state.reshape(147)
       state = state.tolist()
       states.append(state)
elif (env.env.env.metadata['name'] == 'simple_tag_v2'):
    for ag in env.agents:
       state = env.observe(ag)
       state = state.reshape(62)
       state = state.tolist()
       states.append(state)
 else:
    for ag in env.agents:
       state = env.observe(ag).tolist()
       states.append(state)

 s["State"] = states

 step = 0
 done = False

 # Storage for cumulative reward
 cumulativeReward = 0.0

 overSteps = 0
 if s["Mode"] == "Testing":
   image_count = 0

 while not done and step < 500:

  s.update()

  # Printing step information
  if (printStep):  print('[Korali] Frame ' + str(step), end = '')

  actions = s["Action"]
  rewards = []
  for ag in env.agents:
   if s["Mode"] == "Testing" and (env.env.env.metadata['name']== 'waterworld_v3'):
      obs=env.env.env.env.render('rgb_array')
      im = Image.fromarray(obs)
      fname = os.path.join("/scratch/mzeqiri/korali/examples/study.cases/pettingZoo/images/","image_{0}.png".format(image_count))
      im.save(fname)
      image_count += 1

   '''
   #Doesn't work without a monitor, cannot use on panda
   elif s["Mode"] == "Testing" and ( env.env.env.metadata['name']== 'multiwalker_v9'):
      obs = env.env.env.render('rgb_array')
      im = Image.fromarray(obs)
      fname = os.path.join("/scratch/mzeqiri/korali/examples/study.cases/pettingZoo/images_multiwalker/","image_{0}.png".format(image_count))
      im.save(fname)
      image_count += 1
   '''

   observation, reward, done, truncation, info = env.last()
   rewards.append(reward)
   action = actions.pop(0)

   if done and (env.env.env.metadata['name']== 'multiwalker_v9'):
    continue

   if (env.env.env.metadata['name']== 'waterworld_v3') or (env.env.env.metadata['name']== 'multiwalker_v9'):
      env.step(np.array(action,dtype= 'float32'))
   else: # Pursuit
      if done:
         #if persuit is done only action is NONE
         continue
      env.step(action[0])

  # Getting Reward
  s["Reward"] = rewards

  # Storing New State
  states = []
   
  if (env.env.env.metadata['name'] == 'pursuit_v4'):
    for ag in env.agents:
       state = env.observe(ag)
       state = state.reshape(147)
       state = state.tolist()
       states.append(state)
  elif (env.env.env.metadata['name'] == 'simple_tag_v2'):
    for ag in env.agents:
       state = env.observe(ag)
       state = state.reshape(62)
       state = state.tolist()
       states.append(state)
  else:
      for ag in env.agents:
       state = env.observe(ag).tolist()
       states.append(state)

  s["State"] = states

  # Advancing step counter
  step = step + 1

 # Setting termination status
 if (not env.agents):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

 if s["Mode"] == "Testing":
   env.close()

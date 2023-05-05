#!/usr/bin/env python3
#modification: multiwalker_v7 to multiwalker_v9; Add Simple Adversary from MPE environment
import math
import pdb
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import imageio

def initEnvironment(e, envName, multPolicies):

 # Creating environment
 if (envName ==  'Waterworld'):
    from pettingzoo.sisl import waterworld_v4
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
   env = simple_tag_v2.env(num_good=1, num_adversaries=1,continuous_actions=True, render_mode='rgb_array')
   stateVariableCount = [12, 10]
   actionVariableCount = [5, 5]
   ac_upper = 1
   ac_low = 0
   numIndividuals = 2
   agentsPerTeam = [1,1]
 else:
   print("Environment '{}' not recognized! Exit..".format(envName))
   sys.exit()

 if (envName == 'Simpletag'):
    for i in range(len(stateVariableCount)):
       for j in range(stateVariableCount[i]):
          e["Variables"][sum(stateVariableCount[:i]) + j]["Name"] = "State Variable " + str(j)
          e["Variables"][sum(stateVariableCount[:i]) + j]["Type"] = "State"
          e["Variables"][sum(stateVariableCount[:i]) + j]["Team Index"] = i
 else:
    for i in range(stateVariableCount):
       e["Variables"][i]["Name"] = "State Variable " + str(i)
       e["Variables"][i]["Type"] = "State"

 
 if (envName == 'Waterworld') or (envName == 'Multiwalker') or (envName == 'Simpletag'):
   ### Defining problem configuration for continuous environments
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = lambda x : agent(x, env)
    e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
    e["Problem"]["Agents Per Environment"] = numIndividuals
    if (multPolicies == 1) :
       if (envName == 'Simpletag'):
          e["Problem"]["Policies Per Environment"] = len(agentsPerTeam)
       else:
          e["Problem"]["Policies Per Environment"] = numIndividuals
 else:
    ### Defining problem configuration for discrete environments
    e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
    e["Problem"]["Environment Function"] = lambda x : agent(x, env)
    e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
    e["Problem"]["Possible Actions"] = possibleActions
    e["Problem"]["Agents Per Environment"] = numIndividuals
    if (multPolicies == 1) :
       e["Problem"]["Policies Per Environment"] = numIndividuals

 ## Defining the new variable Agents per Team for each environment
 if (envName == 'Simpletag'):
   e["Problem"]["Agents Per Team"] = agentsPerTeam

 if (envName == 'Waterworld') or (envName == 'Multiwalker'):
   # Defining Action Variables
    for i in range(actionVariableCount):
       e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
       e["Variables"][stateVariableCount + i]["Type"] = "Action"
       e["Variables"][stateVariableCount + i]["Lower Bound"] = float(ac_low)
       e["Variables"][stateVariableCount + i]["Upper Bound"] = float(ac_upper)
       e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = math.sqrt(0.2) * (ac_upper - ac_low)
 elif (envName ==  'Pursuit'):
    for i in range(actionVariableCount):
       e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
       e["Variables"][stateVariableCount + i]["Type"] = "Action"
 else:
    for i in range(len(actionVariableCount)):
       for j in range(actionVariableCount[i]):
          e["Variables"][sum(stateVariableCount) + sum(actionVariableCount[:i]) + j]["Name"] = "Action Variable " + str(j)
          e["Variables"][sum(stateVariableCount) + sum(actionVariableCount[:i]) + j]["Type"] = "Action"
          e["Variables"][sum(stateVariableCount) + sum(actionVariableCount[:i]) + j]["Lower Bound"] = float(ac_low)
          e["Variables"][sum(stateVariableCount) + sum(actionVariableCount[:i]) + j]["Upper Bound"] = float(ac_upper)
          e["Variables"][sum(stateVariableCount) + sum(actionVariableCount[:i]) + j]["Initial Exploration Noise"] = math.sqrt(0.2) * (ac_upper - ac_low)
          e["Variables"][sum(stateVariableCount) + sum(actionVariableCount[:i]) + j]["Team Index"] = i

 return numIndividuals

def agent(s, env):

 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False

 env.reset(seed=s["Sample Id"])

 states = []

# add MPE here
 if (env.env.env.metadata['name'] == 'pursuit_v4'):
    for ag in env.agents:
       state = env.observe(ag)
       state = state.reshape(147)
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
 
 if s["Mode"] != "Testing" and env.env.env.metadata['name'] == 'simple_tag_v2':
    renderings_dir = "Renderings"
    if not os.path.exists(renderings_dir):
       os.makedirs(renderings_dir)

 while not done and step < 25 :

    s.update()
  
    # Printing step information    
    if (printStep):  print('[Korali] Frame ' + str(step), end = '')
  
    actions = s["Action"]
    rewards = []
    sampleId = s["Sample Id"]
    generation = sampleId // 10
    for agent_id, ag in enumerate(env.agents):
       if s["Mode"] == "Testing" and (env.env.env.metadata['name']== 'waterworld_v4'):
          obs=env.env.env.env.render('rgb_array')
          im = Image.fromarray(obs)
          fname = os.path.join("/scratch/mzeqiri/korali/examples/study.cases/pettingZoo/images/","image_{0}.png".format(image_count))
          im.save(fname)
          image_count += 1

       observation, reward, done,truncation, info = env.last()
       rewards.append(reward)
       action = actions.pop(0)
     
       if done and (env.env.env.metadata['name']== 'multiwalker_v9'):
          continue

       if (env.env.env.metadata['name']== 'simple_tag_v2'):
          if done or truncation:
             action = None
             continue
             
       if (env.env.env.metadata['name']== 'pursuit_v4'):
          if done:
             #if persuit is done only action is NONE
             continue
          env.step(action[0])
       else: # Pursuit
          env.step(np.array(action,dtype= 'float32'))
    
    # Saving the rendered frames
    if s["Mode"] != "Testing" and env.env.env.metadata['name'] == 'simple_tag_v2' and generation % 100 == 0:
        generation_renderings_dir = os.path.join(renderings_dir, f"generation_{generation}")
        episode_renderings_dir = os.path.join(generation_renderings_dir, f"episode_{sampleId}")
        os.makedirs(episode_renderings_dir, exist_ok=True)
        mpe_env = env.env
        img_array = mpe_env.render()
        img = Image.fromarray(img_array, 'RGB')
        frame_path = os.path.join(episode_renderings_dir, f"frame_{step}.png")
        img.save(frame_path)  

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
    else:
       for ag in env.agents:
          state = env.observe(ag).tolist()
          states.append(state)
  
    s["State"] = states

    # Advancing step counter
    step = step + 1
 
 # Creating video from given frames
 if s["Mode"] != "Testing" and env.env.env.metadata['name'] == 'simple_tag_v2' and generation % 100 == 0:
    episode_frame_files = sorted([f for f in os.listdir(episode_renderings_dir) if f.endswith('.png')],
                             key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    new_size = (704, 704)
    episode_frames = []
    for frame_file in episode_frame_files:
       frame_path = os.path.join(episode_renderings_dir, frame_file)
       img = Image.open(frame_path)
       img_resized = img.resize(new_size)
       episode_frames.append(np.array(img_resized))

    fps = 6  # Adjust the frames per second as needed
    video_file = os.path.join(episode_renderings_dir, 'output_video.mp4')
    imageio.mimsave(video_file, episode_frames, fps=fps)

 # Setting termination status
 if (not env.agents):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

 if s["Mode"] == "Testing":
   env.close()

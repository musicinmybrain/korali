#!/bin/bash

# Install openAI gym

python3 -m pip install gym[mujoco]

# run 5 generations

python3 run-vracer.py --env Swimmer-v4 --dis "Clipped Normal" 
  
  

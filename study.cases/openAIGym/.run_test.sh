#!/bin/bash
exit #TODO: reinstantiate test (DW 25.10.2023)

# Install openAI gym

python3 -m pip install gym[mujoco]
python3 -m pip install cffi

##### Running Test

# delete old test
rm -rf _result_vracer_HalfCheetah-v4_-1/

echo "Running ${file} ..."
python3 run-vracer.py --env HalfCheetah-v4 --exp 1000 --run -1

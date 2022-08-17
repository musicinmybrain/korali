#!/bin/bash

set -x

pip install --upgrade --quiet phiflow
pip install --upgrade --quiet tensorflow
pip install --quiet stable-baselines3==1.1
# Maybe stable-baselines3 won't be needed in the end. It's just here for run-burgers-training-test.py

#python check_phiflow_version.py

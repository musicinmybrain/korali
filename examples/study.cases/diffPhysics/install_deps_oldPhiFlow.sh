# THIS VERSION WILL INSTALL AN OUTDATED VERSION OF PHIFLOW
# ONLY USE THIS VERSION IF YOU WANT TO EXECUTE run-burger-training.py

#!/bin/bash

set -x

pip install --quiet phiflow==1.5.1
pip install --upgrade --quiet tensorflow
#pip install --quiet stable-baselines3==1.1

#python check_phiflow_version.py

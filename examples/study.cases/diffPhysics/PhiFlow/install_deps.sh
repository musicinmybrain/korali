#!/bin/bash

set -x

pip install --upgrade --quiet phiflow
pip install --upgrade --quiet tensorflow
# the following is only used for the DL example
pip install h5py==2.6.0

#python check_phiflow_version.py

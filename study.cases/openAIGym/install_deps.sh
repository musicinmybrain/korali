#!/bin/bash

if [ ! -d _model/pyBulletEnvironments ]; then
 mkdir _model/pyBulletEnvironments
 git clone https://github.com/bulletphysics/bullet3.git
 cp bullet3/examples/pybullet/gym/pybullet_envs/* _model/pyBulletEnvironments
 rm -rf ./bullet3/ _model/pyBulletEnvironments/kerasrl_utils.py
fi

python3 -m pip install pybullet

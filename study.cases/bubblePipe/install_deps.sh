#!/bin/bash

baseDir=$PWD

# Removing and recloning
rm -rf _deps
git clone git@github.com:cselab/aphros-dev.git _deps/aphros_src --recursive

# Deployment phase
pushd _deps/aphros_src/deploy
./install_setenv --profile daint ${baseDir}/_deps/aphros
. ap.setenv
mkdir build
cd build
cmake ..
make -j6
make install
popd

# Compiling Source
pushd _deps/aphros_src/src
. ap.setenv
make -j6
make install
popd

# Obtaining base simulation config
rm -rf _config
mkdir _config
cp -r _deps/aphros_src/sim/sim33_epflopt/case/pipe/* _config
rm _config/vis/vis.py
cp _model/run.sh _config
cp  _deps/aphros_src/sim/sim33_epflopt/case/korali/vis/vis.py _config/vis/


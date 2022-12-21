#!/bin/bash

rm -rf _deps
git clone git@github.com:cselab/msode.git _deps/msode --recursive
mkdir _deps/msode/build
pushd _deps/msode/build
cmake .. -DUSE_SMARTIES=OFF -DBUILD_TESTING=OFF -DBUILD_WITH_CONTRACTS=OFF -DCMAKE_CXX_FLAGS=-w -DENABLE_STACKTRACE=OFF
make -j6
popd

rm -rf _config
cp -r _deps/msode/launch_scripts/rl/config _config


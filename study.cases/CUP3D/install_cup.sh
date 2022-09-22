#!/bin/bash

rm -rf _deps

# clone CUP
git clone -b AMR --recursive git@gitlab.ethz.ch:mavt-cse/CubismUP_3D.git _deps/CUP-3D

make -C _deps/CUP-3D/makefiles -j


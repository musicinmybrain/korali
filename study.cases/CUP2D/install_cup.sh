#!/bin/bash

rm -rf _deps

# clone / compile CUP2D
git clone -b amr-mpi --recursive git@github.com:cselab/CUP2D.git _deps/CUP-2D
make gpu=true -C _deps/CUP-2D/makefiles -j

# clone / compile CUP3D
# git clone -b AMR --recursive git@github.com:cselab/CUP3D.git _deps/CUP-3D
# make -C _deps/CUP-3D/makefiles -j

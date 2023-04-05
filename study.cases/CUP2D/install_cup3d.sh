#!/bin/bash

# clone / compile CUP3D
git clone -b AMR --recursive git@github.com:cselab/CUP3D.git _deps/CUP-3D
make gpu=true -C _deps/CUP-3D/makefiles -j

#!/bin/bash

./run-ddpg -poissonType cosine -muteAll 1 -bpdx 32 -bpdy 16 -tdump 0 -nu 0.000018 -tend 0 -shapes 'halfDisk_radius=.06_angle=20_xpos=.2_bForced=1_bFixed=1_xvel=0.15_tAccel=5,stefanfish_L=.2_xpos=.5'

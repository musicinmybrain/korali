#!/bin/bash

# Defaults for Options
BPDX=${BPDX:-2}
BPDY=${BPDY:-2}
LEVELS=${LEVELS:-5}
RTOL=${RTOL-1}
CTOL=${CTOL-0.1}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.2}
DT=${DT:-1e-2}
PT=${PT:-1e-5}
PTR=${PTR:-1e-4}
NU=${NU:-0.00001}

RADIUS=${RADIUS:-0.2}

# Settings for simulation
echo "###############################"
echo "setting simulation options"
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1 -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 0"
echo $OPTIONS
echo "----------------------------"
# Setting for obstacle
echo "setting obstacle options"
OBJECTS="smartDisk radius=$RADIUS"
echo $OBJECTS
echo "###############################"

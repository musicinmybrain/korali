#!/bin/bash
# Defaults for Options
BPDX=${BPDX:-3}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-4}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-1.4}
CFL=${CFL:-0.22}
# Defaults for Objects
XPOS=${XPOS:-0.2}

YPOS1=${YPOS:-0.6}
YPOS2=${YPOS2:-0.8}

XVEL=${XVEL:-0.15}
#XVEL=${XVEL:-0.3}

MAAXIS=${MAAXIS:-0.0405}
MIAXIS=${MIAXIS:-0.0135}

#NU=${NU:-0.0001215}
NU=${NU:-0.000243}

# bBlockAng=1 means we do not compute the pressure forces applied on the fans

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.0 -nu $NU -poissonTol 1.0e-5 -tend 0 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver cuda_iterative"

OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=1
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=1"
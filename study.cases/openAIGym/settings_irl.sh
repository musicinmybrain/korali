# Defaults for Options
ENV=${ENV:-Swimmer-v4}  # environment
EBRU=${EBRU:-100}       # experiences between reward updates
DBS=${DBS:-10}          # demonstration batch size
BBS=${BBS:-50}          # background batch size
BSS=${BSS:-2000}        # background sample size
EXP=${EXP:-5000000}     # number of experiences
RNN=${RNN:-16}          # reward neural net size
POL=${POL:-Quadratic}   # demo policy type
RUN=${RUN:-0}           # run tag

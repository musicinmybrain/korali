#!/bin/bash

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $(($N+1)) -c 12"
fi

set -x

$mpiflags ./run-vracer-transport -extent 4.0 -bpdx 8 -bpdy 8 -levelMax 4 -Rtol 0.1 -Ctol 0.01 -muteAll 1 -verbose 0 -tdump 0 -nu 0.00001 -tend 0 -shapes 'smartDisk_radius=.10'


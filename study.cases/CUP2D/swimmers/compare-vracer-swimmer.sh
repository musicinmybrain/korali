#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./compare-vracer-swimmer.sh RUNNAME TASK"
	exit 1
fi

RUNNAME=$1
TASK=$2

# path to previous run
RESULTSPATH=/scratch/snx3000/pweber/korali/2D/_trainingResults

# number of agents
NAGENTS=100

# number of nodes per worker
NRANKS=9

# number of cores per worker
NUMCORES=12

# get number of available nodes (should be equal to NRANKS+1)
N=$SLURM_NNODES

# create runfolder
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp compare-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

set -ux

# $mpiflags ./eval-vracer-swimmer ${OPTIONS} -shapes "${OBJECTS}" -nAgents $NAGENTS
srun --nodes=$((N-1)) --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./compare-vracer-swimmer -resultsPath $RESULTSPATH -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./compare-vracer-swimmer -resultsPath $RESULTSPATH -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))
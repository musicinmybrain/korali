#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-train.sh RUNNAME DIMENSION"
	exit 1
fi

RUNNAME=$1
DIMENSION=$2
EXECUTABLE=train-swimmer-"${DIMENSION}"D

NWORKER=63  # number of workers
NRANKS=1    # nodes per worker
NUMCORES=12 # cores per worker
NNODES=$(( $NWORKER * $NRANKS )) # number of workers * number of nodes per worker

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp $EXECUTABLE ${RUNPATH}
cp *.cpp *.hpp ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch_training-"${DIMENSION}"D
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((NNODES+1))
##SBATCH --dependency=afterany: [ID]


srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./$EXECUTABLE $EXTRA -task $TASK -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./$EXECUTABLE -nRanks $(( $NRANKS * $NUMCORES ))
EOF

chmod 755 daint_sbatch_training-"${DIMENSION}"D
sbatch daint_sbatch_training-"${DIMENSION}"D

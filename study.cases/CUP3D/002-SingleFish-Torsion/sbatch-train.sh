#! /usr/bin/env bash

if [ $# -lt 5 ] ; then
	echo "Usage: ./sbatch-train.sh RUNNAME DIMENSION TRAIN NODES NODES_PER_SAMPLE"
	echo "       RUNNAME           = name of the directory where the run is performed "
	echo "       DIMENSION         = 2 or 3"
	echo "       TRAIN             = 1 (for training) or 0 (for testing)"
	echo "       NODES             = total number of nodes to use (not counting +1 for Korali)"
	echo "       NODES_PER_SAMPLE  = number of nodes per RL sample. CAREFUL: NODES mod NODES_PER_SAMPLE must be 0!"
	exit 1
fi

if [ $(( $4 % $5 )) -ne 0 ]
then
	echo "NODES % NODES_PER_SAMPLE must be zero."
	exit 1
fi

RUNNAME=$1
DIMENSION=$2
TRAIN=$3
NODES=$4
NODES_PER_SAMPLE=$5

if [ $DIMENSION == 2 ] #Use GPU solver with 1 MPI rank & 12 threads per node
then
	THREADS=12
	RANKS=1
else                   #Use CPU solver with 12 MPI ranks & 1 thread per node
	THREADS=1
	RANKS=12
fi
RANKS_PER_SAMPLE=$(( $NODES_PER_SAMPLE * $RANKS ))

if [ $TRAIN == 1 ]
then
EXECUTABLE=train-swimmer-"${DIMENSION}"D
BATCH_FILE=daint_sbatch_training-"${DIMENSION}"D
CLOCK=${CLOCK:-24:00:00}
PARTITION=${PARTITION:-normal}
else
EXECUTABLE=test-swimmer-"${DIMENSION}"D
BATCH_FILE=daint_sbatch_testing-"${DIMENSION}"D
CLOCK=${CLOCK:-00:30:00}
PARTITION=${PARTITION:-debug}
fi


# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp $EXECUTABLE ${RUNPATH}
cp *.cpp *.hpp ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >${BATCH_FILE}
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=${CLOCK}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=$(($NODES + 1))
##SBATCH --dependency=afterany:38305561

srun --nodes=$NODES --ntasks-per-node=$RANKS --cpus-per-task=$THREADS --threads-per-core=1 ./$EXECUTABLE -nRanks $RANKS_PER_SAMPLE : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./$EXECUTABLE -nRanks $RANKS_PER_SAMPLE

EOF

chmod 755 ${BATCH_FILE}
sbatch ${BATCH_FILE}

#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-cylinder.sh RUNNAME REGULARIZER"
	exit 1
fi

RUNNAME=$1
NWORKER=1
NRANKS=3
NUMCORES=128
NNODES=$(( $NWORKER * $NRANKS ))
REG=$2
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-cylinder ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >lumi_sbatch_testing
#!/bin/bash -l
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=00:30:00
#SBATCH --partition=small
#SBATCH --nodes=$((NNODES)) --ntasks-per-node=128 --cpus-per-task=1 --hint=nomultithread
#SBATCH hetjob
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
srun --het-group=0,1 ./eval-vracer-cylinder -nRanks $(( $NRANKS * $NUMCORES )) -reg $2
EOF
chmod 755 lumi_sbatch_testing
sbatch lumi_sbatch_testing

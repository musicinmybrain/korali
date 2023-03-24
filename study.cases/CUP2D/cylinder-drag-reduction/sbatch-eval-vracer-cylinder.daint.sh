#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-eval-vracer-cylinder.sh RUNNAME REGULARIZER"
	exit 1
fi

RUNNAME=$1
NNODES=4
NRANKS=48 #=4*12
REG=$2

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-cylinder ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch_testing
#!/bin/bash -l
#SBATCH --account=s1160
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=$((NNODES+1))
srun --nodes=$NNODES --ntasks-per-node=12 --cpus-per-task=1 ./eval-vracer-cylinder -nRanks $NRANKS -reg $REG : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./eval-vracer-cylinder -nRanks $NRANKS -reg $REG
EOF

chmod 755 daint_sbatch_testing
sbatch daint_sbatch_testing

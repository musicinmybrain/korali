#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-naca.sh RUNNAME "
	exit 1
fi

RUNNAME=$1
NNODES=3
NRANKS=384 #=3*128

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-naca ${RUNPATH}
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
export SRUN_CPUS_PER_TASK=8 
srun --het-group=0,1 ./eval-vracer-naca -nRanks $NRANKS
EOF
chmod 755 lumi_sbatch_testing
sbatch lumi_sbatch_testing

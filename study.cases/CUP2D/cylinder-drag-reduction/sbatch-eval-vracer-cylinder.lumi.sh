#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-cylinder.sh RUNNAME "
	exit 1
fi

RUNNAME=$1

#testing is done for 15 cases (for five Re values and three max actuator velocity values)
#we have 1 node / case + 1 node for Korali = 16 nodes

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-cylinder ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >lumi_sbatch_testing
#!/bin/bash -l
##SBATCH --dependency=afterany:3529730
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=128
srun ./eval-vracer-cylinder
EOF
chmod 755 lumi_sbatch_testing
sbatch lumi_sbatch_testing

#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-cylinder.sh RUNNAME "
	exit 1
fi

RUNNAME=$1

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-cylinder3D ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >lumi_sbatch_testing
#!/bin/bash -l
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=128
srun ./eval-vracer-cylinder3D
EOF
chmod 755 lumi_sbatch_testing
sbatch lumi_sbatch_testing

#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-run-vracer-cylinder.sh [name of run] [number of workers] "
	exit 1
fi
RUNNAME=$1
NNODES=$2

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-cylinder ${RUNPATH}
cp run-vracer-cylinder.cpp ${RUNPATH}
cp -r _model ${RUNPATH}
cp -r ../_deps/CUP-2D ${RUNPATH}

cd ${RUNPATH}

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s1160
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((NNODES+1))
srun --nodes=$NNODES --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./run-vracer-cylinder -nRanks 1 : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./run-vracer-cylinder -nRanks 1

EOF

chmod 755 daint_sbatch
sbatch daint_sbatch

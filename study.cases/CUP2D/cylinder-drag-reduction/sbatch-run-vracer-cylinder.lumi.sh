#! /usr/bin/env bash
if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-run-vracer-cylinder.sh [name of run] [number of workers] "
	exit 1
fi
RUNNAME=$1

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-cylinder ${RUNPATH}
cp run-vracer-cylinder.cpp ${RUNPATH}
cp -r _model ${RUNPATH}
cp -r ../_deps/CUP-2D ${RUNPATH}

cd ${RUNPATH}

cat <<EOF >lumi_sbatch
#!/bin/bash -l
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --nodes=16 --ntasks-per-node=128 --cpus-per-task=1 --hint=nomultithread
srun ./run-vracer-cylinder -nRanks 23
EOF

chmod 755 lumi_sbatch
sbatch lumi_sbatch

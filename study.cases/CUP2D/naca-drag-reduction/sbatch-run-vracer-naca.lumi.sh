#! /usr/bin/env bash
if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-run-vracer-naca.sh [name of run] [number of workers] "
	exit 1
fi
RUNNAME=$1
NODES=$2

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-naca ${RUNPATH}
cp run-vracer-naca.cpp ${RUNPATH}
cp -r _model ${RUNPATH}
cp -r ../_deps/CUP-2D ${RUNPATH}

cd ${RUNPATH}

cat <<EOF >lumi_sbatch
#!/bin/bash -l
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=08:00:00
#SBATCH --partition=standard
#SBATCH --nodes=$((NODES+1)) --ntasks-per-node=128 --cpus-per-task=1 --hint=nomultithread
#SBATCH hetjob
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
export SRUN_CPUS_PER_TASK=8 
srun --het-group=0,1 ./run-vracer-naca -nRanks 128
EOF

chmod 755 lumi_sbatch
sbatch lumi_sbatch

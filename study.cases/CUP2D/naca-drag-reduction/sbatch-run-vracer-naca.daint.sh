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

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s1160
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((NODES+1))
srun --nodes=$NODES --ntasks-per-node=12 --cpus-per-task=1 --threads-per-core=1 ./run-vracer-naca -nRanks 12 : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./run-vracer-naca -nRanks 12
EOF
chmod 755 daint_sbatch
sbatch daint_sbatch
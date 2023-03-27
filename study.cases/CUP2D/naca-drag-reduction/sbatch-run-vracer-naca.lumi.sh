#! /usr/bin/env bash
if [ $# -lt 3 ] ; then
	echo "Usage: ./sbatch-run-vracer-naca.sh [name of run] [number of workers] [number of nodes per worker]"
	exit 1
fi
RUNNAME=$1
NWORKER=$2
NRANKS=$3
NNODES=$(( $NWORKER * $NRANKS ))
NUMCORES=128

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
#SBATCH --nodes=$((NNODES)) --ntasks-per-node=128 --cpus-per-task=1 --hint=nomultithread
#SBATCH hetjob
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
srun --het-group=0,1 ./run-vracer-naca -task 0 -nAgents 1 -nRanks $(( $NRANKS * $NUMCORES ))
EOF

chmod 755 lumi_sbatch
sbatch lumi_sbatch

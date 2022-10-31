#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-compare-vracer-swimmer.sh RUNNAME TASK"
	exit 1
fi

RUNNAME=$1
TASK=$2

# path to previous run
RESULTSPATH=/scratch/project_465000158/paweber/korali/100swimmer-old

# number of agents
NAGENTS=100

# number of workers
NWORKER=64

# number of nodes per worker
NRANKS=1

# number of cores per worker
NUMCORES=128

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp compare-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --nodes=$((NNODES)) --ntasks-per-node=${NUMCORES} --cpus-per-task=1 --hint=nomultithread
#SBATCH hetjob
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8

srun  --het-group=0,1 ./compare-vracer-swimmer -resultsPath $RESULTSPATH -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))
EOF

echo "Starting task ${TASK} with ${NWORKER} simulations each using ${NRANKS} ranks with ${NUMCORES} cores"
echo "----------------------------"

chmod 755 daint_sbatch
sbatch daint_sbatch

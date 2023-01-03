#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-swimmer.sh RUNNAME"
	exit 1
fi

RUNNAME=$1
TASK=$2

EVAL=1

# number of agents
NAGENTS=4

# number of evaluation runs
NWORKER=20

# number of nodes per worker
NRANKS=1

# number of cores per worker
NUMCORES=128

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >lumi_sbatch
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

srun  --het-group=0,1 ./eval-vracer-swimmer -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))
EOF

echo "----------------------------"
echo "Starting task ${TASK} with ${NWORKER} simulations each using ${NRANKS} ranks with ${NUMCORES} cores"

chmod 755 lumi_sbatch
sbatch lumi_sbatch

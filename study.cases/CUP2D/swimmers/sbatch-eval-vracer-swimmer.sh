#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-eval-vracer-swimmer.sh RUNNAME TASK"
	exit 1
fi

RUNNAME=$1
TASK=$2

# number of agents
NAGENTS=4

# number of evaluation runs
NWORKER=1

# number of nodes per worker
NRANKS=4

# number of cores per worker
NUMCORES=12

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch_$EVAL
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=normal
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=$((NNODES+1))

srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./eval-vracer-swimmer -eval $EVAL -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./eval-vracer-swimmer -eval $EVAL -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))
EOF

echo "----------------------------"
echo "Starting task ${TASK} with ${NWORKER} simulations each using ${NRANKS} ranks with ${NUMCORES} cores"

chmod 755 daint_sbatch_$EVAL
sbatch daint_sbatch_$EVAL

## FOR GPU
# srun --nodes=$NNODES --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./eval-vracer-swimmer -eval $EVAL -task $TASK -nAgents $NAGENTS -nRanks $NRANKS : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./eval-vracer-swimmer -eval $EVAL -task $TASK -nAgents $NAGENTS -nRanks $NRANKS

## FOR PURE MPI
# srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./eval-vracer-swimmer -eval $EVAL -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./eval-vracer-swimmer -eval $EVAL -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))

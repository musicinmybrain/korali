#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-vracer-windmill.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of workers/simulations in parallel
NWORKER=16

# number of nodes per worker/simulation
NRANKS=1

# number of cores per nodes (for workers)
NUMCORES=12

# number of worker * number of nodes per worker = number of nodes in total
NNODES=$(($NWORKER * $NRANKS))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-windmill ${RUNPATH}
cp settings.sh ${RUNPATH}
cp avgprofiles/avgprofiles.dat ${RUNPATH}/avgprofiles.dat
cp avgprofiles/stdprofiles.dat ${RUNPATH}/stdprofiles.dat
cd ${RUNPATH}

# STATE:
# 1 = {omega1, omega2}
# 2 = velocity profile
# 3 = velocity profile + {omega1, omega2}

# REWARD:
# 1 = squared difference between deviation from the mean 
# 	  at time t and time t-1, normalized by the target profile
# 2 = squared difference between deviation from the mean 
# 	  at time t and time t-1, non-normalized
# 3 = log-likelihood for hypothetical normal distribution of profiles
# 4 = slight reward given, goal to teach agent to have angular velocity 
# 	  smaller than 10. 

# ALPHA:
# index of the simulation we wish to learn from,
# values between 0 and 10. 11 means we use all of them

# SEQLEN:
# the sequence length to be used for the RNNs

STATE=1
REWARD=5
ALPHA=6
SEQLEN=20

# source settings.sh

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((NNODES+1))

srun --nodes=$NNODES --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS ))

EOF

# gpu
# srun --nodes=$NNODES --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS ))

# mpi 
# srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1  ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -state $(($STATE)) -reward $(($REWARD)) -alpha $(($ALPHA)) -seqLen $(($SEQLEN)) -nRanks $(( $NRANKS * $NUMCORES ))


# srun --nodes=$NNODES --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1  ./run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}" -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-vracer-windmill -nRanks $(( $NRANKS * $NUMCORES ))
# srun ./run-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}"
# srun ./eval-vracer-windmill ${OPTIONS} -shapes "${OBJECTS}"

echo "Starting ${NWORKER} simulations each using ${NRANKS} nodes with ${NUMCORES} cores"
echo "----------------------------"

chmod 755 daint_sbatch
sbatch daint_sbatch
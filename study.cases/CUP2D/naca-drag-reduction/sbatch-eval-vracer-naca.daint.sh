#! /usr/bin/env bash
if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-naca.sh RUNNAME "
	exit 1
fi
RUNNAME=$1
NODES=4
NRANKS=48 # = NODES * 12

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-naca ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch_testing
#!/bin/bash -l
#SBATCH --account=s1160
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=$((NODES+1))
srun --nodes=$NODES --ntasks-per-node=12 --cpus-per-task=1 --threads-per-core=1 ./eval-vracer-naca -nRanks $NRANKS : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./eval-vracer-naca -nRanks $NRANKS
EOF
chmod 755 daint_sbatch_testing
sbatch daint_sbatch_testing
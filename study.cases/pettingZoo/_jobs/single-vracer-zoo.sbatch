#!/bin/bash -l
#SBATCH --job-name="zoo_VRACER"
#SBATCH --output=zoo_VRACER_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Choose Environment
env=Multiwalker
#env=Waterworld

# Choose model
model=0
#model=1

# Choose Policy Distribution
dis="Clipped Normal"
#dis="Normal"
#dis="Squashed Normal"
#dis="Truncated Normal"

# Set l2 regularization
l2=0.0

# Set off policy target
opt=0.1

# Set learning rate
lr=0.0001

# Set nn width
nn=128

pushd ..

cat run-vracer.py

# Copy relevant files to RUNPATH
RUNPATH=${SCRATCH}/pettingZoo/${ENV}/\$SLURM_JOB_ID
mkdir -p $RUNPATH
cp run-vracer.py $RUNPATH
cp -r _model $RUNPATH

popd

pushd $RUNPATH

# Run RL
OMP_NUM_THREADS=12 python3 run-vracer.py --env $env --dis "$dis" --l2 $l2 --opt $opt --lr $lr --nn $nn --model $model --run 0

cd results

# Grab result dir
resdir=$(ls -d _result_vracer_*)

# Plot learning
python3 -m korali.rlview --dir $resdir --output vracer.png

popd

date

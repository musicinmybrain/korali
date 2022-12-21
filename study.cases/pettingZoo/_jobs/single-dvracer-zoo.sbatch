#!/bin/bash -l
#SBATCH --job-name="zoo_DVRACER"
#SBATCH --output=zoo_DVRACER_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Choose Environment
env=Pursuit

# Choose model
model=0
#model=1

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
cp run-dvracer.py $RUNPATH
cp -r _model $RUNPATH

popd

pushd $RUNPATH

# Run RL
OMP_NUM_THREADS=12 python3 run-dvracer.py --env $env --l2 $l2 --opt $opt --lr $lr --n $nn --model $model --run 0

cd results

# Grab result dir
resdir=$(ls -d _result_dvracer_*)

# Plot learning
python3 -m korali.rlview --dir $resdir --output dvracer.png

popd

date

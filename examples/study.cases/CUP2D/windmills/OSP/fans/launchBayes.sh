#!/bin/bash -l
#
#SBATCH --job-name="bayes"
#SBATCH --time=00:30:00
#SBATCH --nodes=6
#SBATCH --ntasks=61
#SBATCH --output=bayes.out
#SBATCH --error=bayes.err
#SBATCH --account=s929
#SBATCH -C gpu

srun python bayesian_design.py
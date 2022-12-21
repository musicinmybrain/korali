#!/bin/bash -l
#
#SBATCH --job-name="2D_bayes"
#SBATCH --time=05:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --output=test.%j.out
#SBATCH --error=test.%j.err
#SBATCH --account=s929
#SBATCH -C gpu

srun python 2D_bayesian_design.py
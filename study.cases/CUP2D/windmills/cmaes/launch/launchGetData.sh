#!/bin/bash -l
#
#SBATCH --job-name="getData"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --output=getData.out
#SBATCH --error=getData.err
#SBATCH --account=s929
#SBATCH -C gpu

cd ../postprocess
srun python get_data_gridsearch.py
#!/bin/bash -l

#SBATCH --job-name="Render:1024"
#SBATCH --time=08:00:00
#SBATCH --nodes=12
#SBATCH --ntasks=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --account=s929

#SBATCH --array=1-333
#========================================
# load modules
module load daint-gpu ParaView/5.9.1-CrayGNU-20.11-EGL-python3

srun -n $SLURM_NTASKS -N $SLURM_NNODES --cpu_bind=sockets pvbatch script.py --case ${SLURM_ARRAY_TASK_ID} --path "/scratch/snx3000/mchatzim/CubismUP3D/ELLIPSOID/" --name "./SpartanFish"

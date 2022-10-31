#!/bin/bash -l
#SBATCH --job-name="ABF3D_GFPT"
#SBATCH --output=ABF3D_GFPT_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=eth2
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc

date

pushd ..

cat run-gfpt.cpp

expDir=$SCRATCH/ABF3D_GFPT/$SLURM_JOB_ID
mkdir -p $expDir
cp run-gfpt $expDir
cp -r _config $expDir

popd

pushd $expDir

OMP_NUM_THREADS=36 ./run-gfpt 

popd

date

#!/bin/bash -l
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --partition=xfer

command="cp -R "
echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2\n"
srun -n $SLURM_NTASKS $command $1 $2
echo -e "$SLURM_JOB_NAME finished on $(date)\n"

if [ -n "$3" ]; then
 # unset memory constraint enabled on xfer partition
 unset SLURM_MEM_PER_CPU
 # submit job with dependency
 sbatch --dependency=afterok:$SLURM_JOB_ID $3
fi
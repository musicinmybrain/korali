#!/bin/bash -l

RUNPATH=${SCRATCH}/korali/baseline/${ENV}/run${RUN}
mkdir -p $RUNPATH

cp ../run-b-vracer.py $RUNPATH
cp -r ../_model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name="${ENV}_run${RUN}"
#SBATCH --output=${ENV}_run${RUN}_%j.out
#SBATCH --error=${ENV}_run${RUN}_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

OMP_NUM_THREADS=12 python run-b-vracer.py --env "$ENV"
EOF

sbatch run.sbatch

#!/bin/bash -l

source settings.sh

echo "Environment:"         $ENV
echo "Model:"               $MODEL
echo "Policy distribution:" $DIS
echo "L2 Regularizer:"      $L2
echo "Off-policy target:"   $OPT
echo "Learning rate:"       $LR
echo "NN size:"             $NN
echo "RUN"					$RUN
echo "Multi"				$MULTI

RUNPATH=${SCRATCH}/pettingZoo/${ENV}_model_${MODEL}_multi_pol_${MULTI}
mkdir -p $RUNPATH

cp run-vracer.py $RUNPATH
cp -r _model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=zoo_VRACER_${ENV}
#SBATCH --output=zoo_${ENV}_%j.out
#SBATCH --error=zoo_${ENV}_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
python3 run-vracer.py --env "$ENV" --dis "$DIS" --l2 $L2 --opt $OPT --lr $LR --model '$MODEL' --run $RUN --multpolicies $MULTI

export RESDIR=\`ls -d ./results/_result*\`
python3 -m korali.rlview --dir \$RESDIR --out vracer.png --average
EOF

chmod 755 run.sbatch
sbatch run.sbatch

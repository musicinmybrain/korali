#!/bin/bash -l

# create run-folder
FOLDERNAME=${SCRATCH}/korali/ensemble
mkdir -p ${FOLDERNAME}

# move scripts to basefolder
cp ../run-b-vracer.py ${FOLDERNAME}
cp -r ../_model/ ${FOLDERNAME}

# go to base folder
cd ${FOLDERNAME}

# count number of runs
let NUMNODES=0

# Remove existing task file
rm tasks.txt

# Write baseline tasks
for E in Ant-v4 HalfCheetah-v4 Hopper-v4 Humanoid-v4 HumanoidStandup-v4 Reacher-v4 Swimmer-v4 Walker2d-v4;
do 
    for R in {01..05};
    do
        for NP in {02..05};
        do
            RUNFOLDER=${FOLDERNAME}/nPolicies=${NP}/$E
            mkdir -p ${RUNFOLDER}
            cp run-b-vracer.py ${RUNFOLDER}
            cp -r _model/ ${RUNFOLDER}
            cat << EOF >> tasks.txt
[@ ${RUNFOLDER}/ @] python run-b-vracer.py --env $E --run $R --nPolicies $NP > run${R}.txt
EOF
            let NUMNODES++
        done;
    done;
done

# Write and submit sbatch script
cat << EOF > daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --job-name=ensemble
#SBATCH --output=ensemble_out_%j.txt
#SBATCH --error=ensemble_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=${NUMNODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:0,craynetwork:4

module load GREASY

export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export OMP_NUM_THREADS=12

greasy tasks.txt
EOF

sbatch daint_sbatch

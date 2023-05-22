#!/bin/bash -l

# TO PLOT RESULTS
# for i in Ant-v4 HalfCheetah-v4 Hopper-v4 Humanoid-v4 Swimmer-v4 Walker2d-v4; do python -m korali.rlview --dir baseline/${i} langevin/${i}/nHyperparameters=8 langevin/${i}/nHyperparameters=32 langevin/${i}/nHyperparameters=128 langevin/${i}/nHyperparameters=512 --numRuns 10 --showLegend --showObservations --output ${i}.png; done

# create run-folder
FOLDERNAME=${SCRATCH}/korali/UQRL/langevin
mkdir -p ${FOLDERNAME}

# move scripts to basefolder
cp ../run-b-vracer.py ${FOLDERNAME}
cp -r ../_model/ ${FOLDERNAME}

# Write baseline tasks
# Ant-v4 Humanoid-v4 Walker2d-v4
# 8 32 128 
for E in HalfCheetah-v4 Hopper-v4 Swimmer-v4;
do 
    for NSGD in 512 1024;
    do
        # Create runfolder, move needed files, and go to runfolder
        cd ${FOLDERNAME}
        RUNFOLDER=${FOLDERNAME}/$E/nHyperparameters=${NSGD}/
        mkdir -p ${RUNFOLDER}
        cp run-b-vracer.py ${RUNFOLDER}
        cp -r _model/ ${RUNFOLDER}
        cd ${RUNFOLDER}

        # Remove existing task file
        rm tasks.txt

        # Create greasy task script
        let NUMNODES=0
        for R in {00..09};
        do
            cat << EOF >> tasks.txt
[@ ${RUNFOLDER}/ @] python run-b-vracer.py --env $E --run $R --bBayesian 1 --nHyperparameters $NSGD > run${R}.txt
EOF
            let NUMNODES++
        done;

        # Write sbatch script and submit job
        cat << EOF > daint_sbatch
#!/bin/bash -l
#SBATCH --account=s1160
#SBATCH --job-name=langevin
#SBATCH --output=langevin_out_%j.txt
#SBATCH --error=langevin_err_%j.txt
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
    done;
done

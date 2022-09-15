#!/bin/bash -l

# create run-folder
FOLDERNAME=${SCRATCH}/korali/UQRL
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
for E in Swimmer-v4;
do 
    for R in {00..20};
    do 
        RUNFOLDER=${FOLDERNAME}/$E
        mkdir -p ${RUNFOLDER}
        cp run-b-vracer.py ${RUNFOLDER}
        cp -r _model/ ${RUNFOLDER}
        cat << EOF >> tasks.txt
[@ ${RUNFOLDER}/ @] python run-b-vracer.py --env $E --run $R > run${R}.txt
EOF
        let NUMNODES++
    done;
done

# Write and submit sbatch script
cat << EOF > daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --job-name=baseline
#SBATCH --output=baseline_out_%j.txt
#SBATCH --error=baseline_err_%j.txt
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

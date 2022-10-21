# Launch continuous envs
for env in Multiwalker Waterworld
do 
    for model in {0..0}
    do
        for run in {0..0}
        do
            for multi in {0..0}
            do
                export ENV=$env
                export MODEL=$model
                export RUN=$run
                export MULTI=$multi 
                ./sbatch-vracer-zoo.sh
            done
        done
    done
done

# Launch discrete envs
for env in Pursuit
do 
    for model in {0..0}
    do
        for run in {0..0}
        do
            for multi in {0..0}
            do
                export ENV=$env
                export MODEL=$model
                export RUN=$run
                export MULTI=$multi
                ./sbatch-dvracer-zoo.sh 
            done
        done
    done
done

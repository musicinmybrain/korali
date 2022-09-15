for E in Ant-v4 HalfCheetah-v4 Hopper-v4 Humanoid-v4 HumanoidStandup-v4 Reacher-v4 Swimmer-v4 Walker2d-v4;
do 
    for R in {01..10}; #"Truncated Normal"; 
    do 
        export ENV=$E
        export RUN=$R 
        ./sbatch-b-vracer.sh 
    done;
done

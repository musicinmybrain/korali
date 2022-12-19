run=700

export ENV="Swimmer-v4"
export POL="Linear"
export EXP=5000000

for D in 4 16 64
do 
    for B in 4 16 64
    do 
        run=$(($run+1))
        export RUN=$run
        export DBS=$D
        export BBS=$B
        bsub < bsub-vracer-irl.lsf
    done
done

run=800

export ENV="Swimmer-v4"
export POL="Linear"
export EXP=3000000

for D in 16
do 
    for B in 64
    do 
        for rep in {1..3}
        do
            run=$(($run+1))
            export RUN=$run
            export DBS=$D
            export BBS=$B
            bsub < bsub-vracer-irl.lsf
        done
    done
done

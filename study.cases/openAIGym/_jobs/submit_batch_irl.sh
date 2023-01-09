run=1200

export ENV="Swimmer-v4"
export POL="Linear"
export EXP=3000000

#for D in 1 4 16 64
for D in 4 16
do 
    for B in 4 16
#    for B in 1 4 16 64
    do 
        for rep in {1..1}
        do
            run=$(($run+1))
            export RUN=$run
            export DBS=$D
            export BBS=$B
            bsub < bsub-vracer-irl.lsf
        done
    done
done

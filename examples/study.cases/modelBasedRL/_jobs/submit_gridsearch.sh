for E in Swimmer-v2;
do
    for D in "Clipped Normal";
    do
        for layers in 5;
        do
            for units in 300;
            do
                for lr in 0.0005;
                do
                    for batch in 512;
                    do
                        for epoch in 250;
                        do
                            for p in 0.0;
                            do
                                for size in 2500000;
                                do
                                    for ws in 0.75; #1.0
                                    do


                                        export ENV=$E
                                        export DIS="$D"
                                        export LRS=$lr
                                        export BATCH=$batch
                                        export EPOCH=$epoch
                                        export LAYERS=$layers
                                        export UNITS=$units
                                        export P=$p
                                        export SIZE=$size
                                        export WS=$ws
                                        ./sbatch-grid-search-openAI.sh
                                    done;
                                done;
                            done;
                        done;
                    done;
                done;
            done;
        done;
    done;
done

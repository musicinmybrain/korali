for retrain in 2000;
do
    for iniretrain in 500;
    do
        for conf in 0.8000 0.8500 0.8750 0.9000 0.9200 0.9400 0.9600 0.9800 0.9900 0.9990;
        do
            export CONF=$conf
            export INI=$iniretrain
            export RE=$retrain
            export M="ThesisFinalL2reg0Net10Ini${INI}Re${RE}_u1.0_Results_${CONF}_1/"
            ./sbatch-vracer-cartpole-L2reg0.sh
        done;
    done; 
done

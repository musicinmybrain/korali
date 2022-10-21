last=`ls _result_gfpt/trajectories* | tail -n 2 | head -n 1`
python3 _deps/msode/tools/plot_trajectories.py $last

#!/bin/bash

resultDir="_result_gfpt"
movieDir="_movie"

cd ${resultDir}
trajectories=`ls traj*.csv`
cd ..

movieDir="_movie"
rm -rf ${movieDir}
mkdir -p ${movieDir}

N=24
(
for t in $trajectories; do
 ((i=i%N)); ((i++==0)) && wait
 baseFile="${t/.csv/}"; echo "Creating ${movieDir}/${baseFile}.png"; python3 _model/plot_trajectory.py --input ${resultDir}/${baseFile}.csv --output ${movieDir}/${baseFile}.png & 
done
)

wait

ffmpeg -framerate 60 -i ${movieDir}/trajectory%06d.png -c:v libx264 -profile:v high -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf 20 -pix_fmt yuv420p ${movieDir}/movie.mp4

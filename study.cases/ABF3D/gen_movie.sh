#!/bin/bash

baseDir="$PWD"
movieDir="_movie"
trajectories=`ls traj*.dat`
rm -rf ${movieDir}
mkdir -p ${movieDir}

N=24
(
for t in $trajectories; do
 ((i=i%N)); ((i++==0)) && wait
 baseFile="${t/.dat/}"; echo "Creating ${movieDir}/${baseFile}.png"; python3 $baseDir/_deps/msode/tools/plot_trajectories.py ${baseFile}.dat --plot_shade --out ${movieDir}/${baseFile}.png & 
done
)

wait

ffmpeg -framerate 60 -i ${movieDir}/trajectories_%06d.png -c:v libx264 -profile:v high -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf 20 -pix_fmt yuv420p ${movieDir}/movie.mp4

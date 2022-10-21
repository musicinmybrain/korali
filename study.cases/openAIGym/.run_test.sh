#!/bin/bash

###### Check if necessary python modules are installed ######

python3 -m pip install gym[mujoco]
exit_code=$?

./install_deps.sh
exit_code=$(( $exit_code || $? ))

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _result*
exit_code=$(( $exit_code || $? ))

##### Creating test files
 
echo "  + Creating test files..."

rm -rf __test-*
exit_code=$(( $exit_code || $? ))

for file in *.py
do
 sed -e 's%\k.run(e)%e["Solver"]["Termination Criteria"]["Max Generations"] = 20; k.run(e)%g' ${file} > __test-${file}
 exit_code=$(( $exit_code || $? ))
done

##### Running Test

file="__test-run-vracer.py"
echo "Running ${file} ..."
#OMP_NUM_THREADS=4 python3 ${file} --env AntBulletEnv-v0
exit_code=$(( $exit_code || $? ))

#file="__test-genMovie.py"
#echo "Running ${file} ..."
#OMP_NUM_THREADS=4 python3 ${file} --env AntBulletEnv-v0 --input _result_vracer_AntBulletEnv-v0 --output movie; check_result

##### Deleting Tests

rm -rf __test-*; 
exit_code=$(( $exit_code || $? ))
  
  

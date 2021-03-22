#!/bin/bash

###### Check if necessary python modules are installed ######
python3 -m pip show gym
if [ $? -ne 0 ]; then
 echo "[Korali] openAI gym not found, aborting test"
 exit 0
fi

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Creating test files
 
echo "  + Creating test files..."

rm -rf __test-*; check_result

for file in *.py
do
 sed -e 's%\["Max Generations"\] =%["Max Generations"] = 20 #%g' ${file} > __test-${file}; check_result
done

##### Running Test

for file in __test-*.py
do
 echo "Running ${file} ..."
 OMP_NUM_THREADS=4 python3 ${file} --env Ant-v2; check_result
done

##### Deleting Tests

rm -rf __test-*; check_result
  
  

#!/bin/bash

source ../functions.sh

pushd ../../tutorials/

logEcho "[Korali] Beginning plotting tests"

for dir in ./a*/_korali_*/
do
  logEcho "-------------------------------------"
  logEcho " Entering Tutorial: $dir"

  if [ ! -d "$dir/cxx" ]; then
    echo "  + No folder named 'cxx' found inside $dir"
    continue
  fi


  pushd $dir/cxx >> $logFile 2>&1

  log "[Korali] Adding Random Seeds..."
  for file in *.cpp
  do
    resultPath="_result_${file%.*}"
    cat $file | sed -e 's/k.run()/k\[\"General\"\]\[\"Random Seed\"\] = 0xC0FFEE; k.run()/g' \
                    -e 's/k.run()/k\[\"General\"\][\"Results Output\"\][\"Path\"\] = \"'$resultPath'\"; k.run()/g' > tmp
    check_result

    log "[Korali] Replacing File..."
    mv tmp $file
    check_result
  done

  logEcho "  + Compiling Tutorial..."

  make clean >> $logFile 2>&1
  check_result

  make -j >> $logFile 2>&1
  check_result

  log "[Korali] Removing any old result files..."
  rm -rf _korali_results >> $logFile 2>&1
  check_result

  for file in *.cpp
  do
    logEcho "  + Running File: $file..."

    resultPath="_result_${file%.*}"
    rm -rf $resultPath >> $logFile 2>&1
    check_result

    log "[Korali] Running $file..."
    ./"${file%.*}" >> $logFile 2>&1
  done

  popd >> $logFile 2>&1
  logEcho " Plotting results from $dir ..."

  logEcho "-------------------------------------"
  python3 -m korali.plotter --test --dir "${dir}" >> $logFile 2>&1
  check_result

done

popd

#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./run-cmaes-windmill.sh RUNNAME A1 A2 F1 F2"
	exit 1
fi
if [ $# -gt 4 ] ; then
	RUNNAME=$1
    A1=$2
    A2=$3
    F1=$4
    F2=$5

fi

echo "Launching a CMAES simulation in the folder $RUNNAME with initial condition A1 = $A1, A2 = $A2, F1 = $F1, F2 = $F2"

# number of total nodes for all CUP simulations
N=8

# number of cores per nodes for worker/simulation
NUMCORES=12

RUNPATH="${SCRATCH}/korali/cmaes/${RUNNAME}"
mkdir -p ${RUNPATH}
cp ../run-cmaes-windmill ${RUNPATH}
cp ../results/slowdiff/x_profile.dat ${RUNPATH}/x_profile.dat
cp ../results/slowdiff/y_profile.dat ${RUNPATH}/y_profile.dat
cd ${RUNPATH}

POP=$N
MU=$((N/2))
REWARD=1

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((N+1))

srun --nodes=$(($POP)) --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-cmaes-windmill -pop $(($POP)) -mu $(($MU)) -f2 $F2 -f1 $F1 -a2 $A2 -a1 $A1 -reward $(($REWARD)) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./run-cmaes-windmill -pop $(($POP)) -mu $(($MU)) -f2 $F2 -f1 $F1 -a2 $A2 -a1 $A1 -reward $(($REWARD))

EOF

chmod 755 daint_sbatch
sbatch daint_sbatch

# correct
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0.25}
# F2=${F2:-0.5}

# launch 1
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 2
# A1=${A1:-2}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 3
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 4
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 5
# A1=${A1:-3}
# A2=${A2:--2}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 6
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 7
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-1}
# F2=${F2:-0}

# launch 8
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-1}

# launch 9
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-1}

# launch 10
# A1=${A1:-2}
# A2=${A2:--2}
# F1=${F1:-0}
# F2=${F2:-0}

# launch 11 with std 0.01
# A1=${A1:-3}
# A2=${A2:--3}
# F1=${F1:-0}
# F2=${F2:-0}


# this means launches 1, 3, 4, 6 and 11 were the same
# remove the first 11 launches

# launch 11 had a super small std

# Ignore all launches before 



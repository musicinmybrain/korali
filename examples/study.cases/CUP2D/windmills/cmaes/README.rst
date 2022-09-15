Windmills : CMA-ES and Grid Search
==============================================

Before running 
-------------------------------

Before running the code, one should create a few folders, where the results will be stored : 

.. code-block:: bash

    mkdir plots
    mkdir data

Before trying to reproduce some of the following results, one needs to apply some changes to the files in the CUP2D repository.
These changes were saved in two files `gitDiffCubism.txt` (for the submodule Cubism) and `gitDiffCUP.txt`(for the CUP2D repository).

Environment setup
-------------------------------
We consider the following setup : two fans are in a 2D rectangular domain of shape 0.525 x 0.7 (x direction - y direction).
The fans rotate with angular velocities :math:`\omega_i (t) = a_i \sin (2 \pi f_i t)`, with i=1, 2 the index of the fan. A target simulation is run over 60s with parameters `(a_1, a_2, f_1, f_2) = (3, -3, 0.25, 0.5)`.
The velocity profile (x and y - direction) is recorded in 16 regions of a rectangular domain, with lower left vertex `(0.35, 0.175)` and upper right vertex `(0.4375, 0.525)`. The time-averaged velocity profile is computed between time 36 and 60 of the simulation.


Grid Search Experiment Description
-------------------------------

The Grid Search experiment is as follows : 
- We attempt to find the original parameters `(a_1, a_2, f_1, f_2)` through a grid search. In order to make the problem more tangible, we set the parameters to be constant `(a_1, f_1) = (3, 0.25)`. 
- The parameter `a_2` is varied between -3.5 and 3.5 over 0.1 increments (71 discrete values). The parameter `f_2` is varied between -1.5 and 5 over steps of 0.1 (66 discrete values).
In total, 4686 simulations are done. 

The objective function is the MSE between the velocity profiles of these simulations and the target simulation.


Grid Search Experiment Files
-----------------------------

- `launch/gridsearch_greasy.py` : creates the folders for the simulation in scratch, copies the CUP2D `./simulation` executable in the folder, creates task file for greasy launcher. 
- `launch/launchGridSearch.sh` : greasy launcher, uses the task.txt file to schedule the jobs. 

Grid Search running
-----------------------------

Running the Grid Search code is very straight forward. Simply run 

.. code-block:: bash

    python gridsearch_greasy

followed by 

.. code-block:: bash

    sbatch launchTask.sh


Grid Search Post-Processing
-------------------------------

In order to obtain the data from the simulations, i.e., the time-averaged velocity profiles for all the simulations 
(as well as the ratios of the amplitudes and frequencies between the constant fan and the varied fan), one simply needs to go in the `launch/` folder :

.. code-block:: bash

    sbatch launchGridGetData.sh

The output are two data files `ratios.npy` and `final_profiles.npy` that are saved in a folder called `data/`.

One can then produce a contourf/surface plot of the objective function as function of the ratio between the amplitudes and the ratio between the frequencies. This is simply done in the folder `postprocessing/` with : 

.. code-block:: bash

    python landscape_gridsearch.sh

CMA-ES Experiment Description
-------------------------------
The CMAES experiment is as follows : 
- The optimizer attempts to find the parameters `(a_1, a_2, f_1, f_2)`.
- The ojective function is the MSE between the velocity profiles of the sample simulations from CMA-ES and the target simulation.


CMA-ES Experiment Files
-----------------------------
- `run-cmaes-windmill.cpp` : setup for CMA-ES using Korali.
- `../_model/windmillEnvironment.*` : setup files for running the CUP2D environment simulation and computing the objective function.
- `launch/run-cmaes-windmill.sh` : launch a single CMA-ES run, with initial parameter values, for interactive node.
- `launch/sbatch-cmaes-windmill.sh` : launch a single CMA-ES run, with initial parameter values, for sbatch nodes.
- `launch/launchSims.sh` and `launch/launchCMAES.sh` : two files used to launch 10 different initial conditions of for CMA-ES
- `postprocess/get_target_profile.py` : produces the data files containing the target x/y velocity profile.

CMA-ES running
-----------------------------

In order to run the CMAES experiment, one can use any of the launch files specified in the previous section. A few things need to be changed in these files in order for the code to function properly.
First, the target profile files need to be copied to the folder where the executable will run : 

.. code-block:: bash

    cp folder/x_profile.dat ${RUNPATH}/x_profile.dat
    cp folder/y_profile.dat ${RUNPATH}/y_profile.dat

The target profile files can be created by launching a CUP2D simulation with the desired parameters `(a_1, a_2, f_1, f_2)`, 
then using the `get_target_profile.py` script that outputs the target in the folder `data/`.

Secondly, one needs to specify the parameters to give to the optimizer. 

In the case of the interactive and batch allocation, 1 node will be allocated for korali and N-1 for the population size. One needs to make sure, the population size is the right number so that each sample is associated with one node.
The parameters are the population size `POP` (which should be set to N-1 for interactive and N for batch allocation), the number of selected samples `MU` (up to the size of the population), the initial amplitudes `A1, A2` and frequencies `F1, F2` for the sampling of the first generation. 
The standard deviation is set to 0.5 by default. : 

.. code-block:: bash

    POP=$((N-1))
    MU=1
    A1=${A1:-3}
    A2=${A2:--3}
    F1=${F1:-0.25}
    F2=${F2:-0.5}
    REWARD=1

To launch the simulation on an interactive node: 

.. code-block:: bash

    ./run-cmaes-windmill.sh run_name

To launch the simulation using sbatch: 

.. code-block:: bash

    ./sbatch-cmaes-windmill.sh run_name

The output of the simulation will be in the scratch folder at the location specified by the user `RUNPATH="${SCRATCH}/korali/${RUNNAME}"`, with `RUNNAME` the name given during the call `run_name`.

Finally, running 

.. code-block:: bash

    ./launchSims.sh

will launch 10 simulations with 10 different initial conditions for CMAES. 


CMA-ES Post-Processing
-------------------------------

In order to obtain the data from the simulations, simply launch the code : 

.. code-block:: bash

    python get_data_cmaes.py

Additionally, if one wants to compare the velocity profiles for the best parameters found by the different simulations, one needs to run :

.. code-block:: bash

    python cup2d_greasy.py
    sbatch launchTask.sh

followed by 

.. code-block:: bash
    python get_data_cup2d.py
    
Then one can run 

.. code-block:: bash
    python compare_profiles.py

which outputs plots of comparisons between the target profile and the solutions obtained with CMA-ES. 

Furthermore, running 

.. code-block:: bash
    python landscape_cmaes.py

will plot the objective function for the samples of the CMA-ES simulation. This gives a good overview of the locality of the solution. 

Windmills : OSP
==============================================

Before running 
-------------------------------

Before running the code, one should create a few folders, where the results will be stored : 

.. code-block:: bash

    mkdir nonlinear/plots
    mkdir fans/plots
    mkdir fans/data

Before trying to reproduce some of the following results, one needs to apply some changes to the files in the CUP2D repository.
These changes were saved in two files `gitDiffCubism.txt` (for the submodule Cubism) and `gitDiffCUP.txt`(for the CUP2D repository). 
Additionally, one needs to call the make command in the folder `tools/uniform_conversion/`.

Environment setup
-------------------------------
We consider the following setup : two fans are in a 2D rectangular domain of shape 0.525 x 0.7 (x direction - y direction).
The fans rotate with angular velocities :math:`\omega_i (t) = a_i \sin (2 \pi f_i t)`, with i=1, 2 the index of the fan. 
The velocity profile (x and y - direction) is recorded in 16 regions of a rectangular domain, with lower left vertex `(0.35, 0.175)` and upper right vertex `(0.4375, 0.525)`. 
The time-averaged velocity profile is computed between time 36 and 60 of the simulation.

OSP Experiment Description
-------------------------------

The Grid Search experiment is as follows : 
- We attempt to find the best sensor placement based on data gathered from simulations. We run many simulations with different initial parameter values and take snapshots of the simulation grid from time 30 to 60 every 0.5. 
- Using Bayesian statistics theory, we can find the placements in the grid that offer the best information gain about the parameters. 
- We use 4 data types : vorticity, pressure, velocity x and velocity y.

OSP Experiment Files
-----------------------------

- `osp_greasy.py` : creates the folders for the simulation in scratch, copies the CUP2D `./simulation` executable in the folder, creates task file for greasy launcher. 
- `launchSims.sh` : greasy launcher, uses the task.txt file to schedule the jobs. 
- `convert_greasy.py` : converts all the non-uniform grids of data from CUP2D into uniform grids. 
- `get_data.py` : iterates over all the data files and creates a file containing the data for all the simulations and 4 different data type. The output is in the file `data/ordered_data.npy`.
- `bayesian_design.py` : mpi script that computes the utility function for all the points in the grid. The output is in the file `data/results_61.npy`
- `launchBayes.sh` : sbatch launcher for the bayesian_design.py script.
- `plot_utility.py` : plots the utility function for each point of the grid and each type of data. The outputs are in `plots/`.

OSP running
-----------------------------

In order to run the experiment, first one runs

.. code-block:: bash

    python osp_greasy.py
    sbatch launchSims.sh

Wait until all the simulations have run. Then

.. code-block:: bash

    python convert_greasy.py
    sbatch launchSims.sh

It is normal that we use the same launch file again. Then one runs

.. code-block:: bash

    python get_data.py

in order to obtain the data needed for the OSP code to run. 

Finally, 

.. code-block:: bash

    sbatch launchBayes.sh

runs the code over 61 nodes in parallel to increase the speed of computation. 

Then one can plot the results using 

.. code-block:: bash

    python plot_utility.py
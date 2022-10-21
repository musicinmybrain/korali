Study Case: Reinforcement Learning on Artificial Bacterial Flagella (ABF) in 3D
=================================================================================

In this study case, we drive a magnetic field to guide ABF towards a common goal in a 3D space.

For more information, read the following paper: `L. Amoudruz, P. Koumoutsakos, Independent Control of Microswimmers with a Uniform Magnetic Field <https://www.cse-lab.ethz.ch/wp-content/papercite-data/pdf/amoudruz2021a.pdf>`_

Setup
---------------------------

1) Install Microswimmers ODE solver

.. code-block:: bash

   ./install_deps.sh

2) Compile the study case by running:

.. code-block:: bash
   
  make -j6

3) Run the test:

.. code-block:: bash
   
  ./run_test.sh

Producing a movie:
-------------------------

The  following command will read the trajectories from the results folder and create an animation showing the progress of the agent's policy during learning.

.. code-block:: bash

   ./genMovie.sh
   


  

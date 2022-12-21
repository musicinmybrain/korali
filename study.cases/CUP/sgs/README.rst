Study Case: Reinforcement Learning for Subgrid Scale Modelling
===============================================================

Environment for optimizing the energy spectrum produced by an underresolved simulation, assuming a dynamic viscosity model (see _deps/CUP2D/Operators/advDiffSGS{.cpp,.h}).

The **state** is given the values of the velocity field on a Cubism block (8x8x2 values)

The **action** is given by modifying the Smargorinsky constant Cs (8x8 values)

The **reward** is given by the normalized distance from the DNS spectrum.

Running the code
----------------

The DNS data can be created by setting the number of gridpoints N and the value Cs for the static Smagorinsky model by running

.. code-block:: bash

	python run-kolmogorov-flow.py

In ./_model there is a spectrum given for N=64, and Cs=0.0. You can run the RL for this spectrum using

.. code-block:: bash

	 python run-vracer.py

Please use the --help command to see the other options for the scripts.

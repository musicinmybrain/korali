Study Case: Reinforcement Learning for Fluids
==============================================

Dependencies
------------

CubismUP-2D has the following prerequisite libraries:

- MPI, with the $MPICXX enviroment variable defined.
- GSL, with the $GSL_ROOT environment variable defined.
- HDF5, with the $HDF5_ROOT environment variable defined.

On Piz Daint:

.. code-block:: bash

        module load daint-gpu GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-fftw cray-python
	export GSL_ROOT=/apps/dom/UES/jenkins/7.0.UP03/21.09/dom-gpu/software/GSL/2.7-CrayGNU-21.09
	export MPICXX=CC
	export CC=cc
	export CXX=CC
	export OMP_NUM_THREADS=12

On Panda/Falcon:

.. code-block:: bash

  	module load gnu mpich python hdf5
  	export GSL_ROOT=/usr

Setup
-----

Install CubismUP-2D by running:

.. code-block:: bash

   ./install_cup.sh

After installing the flow solver you can go to either of the provided examples and compile them using

.. code-block:: bash

	  cd EXAMPLE
	  make -j

Running
-------

The examples can be run using the provided scripts in the subdirectories. See the README there.

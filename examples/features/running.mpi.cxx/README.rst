Running C++ MPI Applications
=====================================================

In this tutorial we show how a C++ MPI model can be executed with Korali.

For more information on running Korali applications in parallel, see :ref:`Parallel Execution <parallel-execution>`. 
For more information on running Korali on MPI, see :ref:`Distributed Conduit <module-conduit-distributed>`. 

MPI Init
---------------------------

Do not forget to init MPI inside the Korali application:

.. code-block:: cpp

    MPI_Init(&argc, &argv);

Distributed Conduit
---------------------------

Run with the `Distributed` conduit to benefit from parallelized model evaluations.
Note that we need to provide it with the MPI communicator we want to use for this instance of Korali.
Next, we set `Ranks Per Worker` to determine how many MPI ranks will be assigned to each Korali worker. This particular example uses `n` MPI ranks per worker, where `n` is passed by argument.

.. code-block:: cpp

  k.setMPIComm(MPI_COMM_WORLD);
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = n;
    
Profiling
---------------------------
    
In some cases it might be useful to activate Korali's internal profiler to analyze
how efficiently workers executed. To enable it, add the following option:

.. code-block:: cpp

    k["Profiling"]["Detail"] = "Full";
    k["Profiling"]["Frequency"] = 0.5;
    
Computational Model
---------------------------
    
If the computational model requires communication between the MPI ranks, you need to obtain the worker-specific sub-communicator

.. code-block:: cpp

    #include "mpi.h"
    #include <korali.hpp>
    ...
    int myRank, rankCount;
    MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &rankCount);

Note that all MPI ranks shall write results to the sample object.

Run
---------------------------

To launch korali, use the corresponding MPI launcher, with a number of MPI ranks that equals k*n+1, where `k` is the number of Korali workers to use, `n` is the number of MPI Ranks per worker, and 1 MPI rank is assigned to the Korali engine.  
In this example, we launch two workers with 4 ranks each, hence we need 9 MPI ranks. 

.. code-block:: bash

    mpirun -n 9 ./run-cmaes 4


Reaction: Simulating the evolution of reactants
===============================================

In this tutorial we show how to **simulate** reactions using a stochastic simulator method (SSM).

The Problem Type
----------------

We set the type of the problem to `Reaction`

.. code-block:: python

    e["Problem"]["Type"] = "Reaction"


Problem Description
------------------- 

A set of reaction equations can be defined as follows:

.. code-block:: python

    e["Problem"]["Reactions"][0]["Equation"] = "[X1]->Y1"
    e["Problem"]["Reactions"][0]["Rate"] = 0.1

    e["Problem"]["Reactions"][1]["Equation"] = "[X2]+Y1->Y2+Z1"
    e["Problem"]["Reactions"][1]["Rate"] = 0.1

    e["Problem"]["Reactions"][2]["Equation"] = "2 Y1+Y2->3 Y1"
    e["Problem"]["Reactions"][2]["Rate"] = 5e-5

    e["Problem"]["Reactions"][3]["Equation"] = "Y1->Z2"
    e["Problem"]["Reactions"][3]["Rate"] = 5.

Variables declared with enclosing square brackets, e.g. [X1], are considered reservoirs and remain unchanged during the simulation.

The Variables
-------------

Each reactant name must be declared as a variable with its initial number of reactants.

.. code-block:: python

    e["Variables"][0]["Name"] = "[X1]"
    e["Variables"][0]["Initial Reactant Number"] = 50000

    e["Variables"][1]["Name"] = "[X2]"
    e["Variables"][1]["Initial Reactant Number"] = 500

    e["Variables"][2]["Name"] = "Y1"
    e["Variables"][2]["Initial Reactant Number"] = 1000

    e["Variables"][3]["Name"] = "Y2"
    e["Variables"][3]["Initial Reactant Number"] = 2000

    e["Variables"][4]["Name"] = "Z1"
    e["Variables"][4]["Initial Reactant Number"] = 0

    e["Variables"][5]["Name"] = "Z2"
    e["Variables"][5]["Initial Reactant Number"] = 0


The Solver
----------

The solver method, the simulation length, the number of simulated trajectories and solver specific configurations are defined in the solver section of the korali application

.. code-block:: python

    e["Solver"]["Type"] = "SSM/SSA"
    e["Solver"]["Simulation Length"] = 20.
    e["Solver"]["Simulations Per Generation"] = 100
    e["Solver"]["Termination Criteria"]["Max Num Simulations"] = 1000
    e["Solver"]["Diagnostics"]["Num Bins"] = 500

For a detailed description of available solver settings see the `SSA <https://korali.readthedocs.io/en/master/modules/solver/SSM/SSA/SSA.html>`_ or `TauLeaping <https://korali.readthedocs.io/en/master/modules/solver/optimizer/SSM/TauLeaping/TauLeaping.html>`_ documentation.

Output
------

The output directory and the number of output files can be configured as

.. code-block:: python

    e["File Output"]["Enabled"] = True
    e["File Output"]["Path"] = '_korali_results'
    e["File Output"]["Frequency"] = 1


Plotting
--------

You can see the averaged trajectories of the SSM by running the command (trajectories are averaged in bins that have been previously defined)

.. code-block:: console
    
    python3 -m korali.plot --dir _korali_results

Integration
==================

In this tutorial we show how to **integrate** a given function. 

Problem Description
------------------- 

We are given the function :math:`f(x,y,z)=x^2+y^2+z^2` for :math:`x,y,z\in [0,1]^3`.
We want to find the integral of this function over its domain.

The Integrand
----------------------

Create a folder named `model`. Inside, create a file with name `integrands.py` and paste the following code,

.. code-block:: python

        def integrand( sample ):
          x = sample["Parameters"][0] 
          y = sample["Parameters"][1] 
          z = sample["Parameters"][2]
          sample["Evaluation"] = x**2+y**2+z**2

This is the function we want to integrate.

The Problem Type
----------------

Then, we set the type of the problem to `Integration`, set the function to integrate and chose the integration method

.. code-block:: python

        e["Problem"]["Type"] = "Integration"
        e["Problem"]["Integrand"] = lambda s : integrand(s)

The Variables
-------------

In this problem there is three variables, `X`, `Y` and `Z`, whose domain we set to [0,1] and in case of Monte Carlo Integration assume an uniform distribution. Furthermore we assume 11 or 9 samples per dimension. Remember that the number of gridpoints must be odd for the Simpson method.

.. code-block:: python

        e["Variables"][0]["Name"] = "x"
        e["Variables"][0]["Number Of Gridpoints"] = 11
        e["Variables"][0]["Lower Bound"] = 0.0
        e["Variables"][0]["Upper Bound"] = 1.0
        
        e["Variables"][1]["Name"] = "y"
        e["Variables"][1]["Lower Bound"] = 0.0
        e["Variables"][1]["Upper Bound"] = 1.0
        e["Variables"][1]["Number Of Gridpoints"] = 9
        
        e["Variables"][2]["Name"] = "z"
        e["Variables"][2]["Lower Bound"] = 0.0
        e["Variables"][2]["Upper Bound"] = 1.0
        e["Variables"][2]["Number Of Gridpoints"] = 11


The Solver
----------
We choose the solver `Integrator`, don't set the execution per generation, to have the summation be performed in one generation,

.. code-block:: python

        e["Solver"]["Type"] = "Integrator/Quadrature"
        e["Solver"]["Method"] = "Simpson"

For a detailed description of Integrator settings see :ref:`Integrator <module-solver-integrator>`.

Finally, we need to add a call to the run() routine to start the Korali engine.

.. code-block:: python

    k.run(e)

Running
-------

We are now ready to run our example:
`python3 ./run-quadrature-integration.py`

The results are saved in the folder `_korali_result/`.

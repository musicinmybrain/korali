Solving Burgers' Equation with JAX
==================================

This project is part of a semester thesis that tries to solve Burgers' Equation with DL methods (Deep RL methods have been planned as well but they have not been optimized).

Run the run-burger-nn.py script for the JAX version

Run the run-vracer-burger.py or run-vracer-burger-jax.py scripts for the Korali version

In the subdirectory PhiFlow, you can find the failed attempts that use the PhiFlow / TensorFlow package

Some of the code has been copied from other sources. Those sources are listed below:

Everything that uses Korali is based on the following project (some scripts have been adapted):
https://github.com/wadaniel/marlpde

The PhiFlow approaches are based on the Physics-based Deep Learning Book (v0.2):
https://physicsbaseddeeplearning.org/intro.html

The JAX version uses some of the Korali scripts as baseline but the main contribution is original work that is based on the example in the JAX documentation:
https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

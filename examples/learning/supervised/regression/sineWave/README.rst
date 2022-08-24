===========
 Sine Wave
===========

These are simple examples of neural networks to approximate sine wave like functions:

- run-ffn.py approximate a function without a time-dependence:
    A simple ffnn tries to learn a function for a training set of (x, y) where the y's are the labels.

    .. math::
        y(x)=\tanh(\exp(\sin(\texttt{trainingInputSet})))\cdot\texttt{scaling}

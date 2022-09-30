Fast Gradient Basd Optimizers
#################################################################

## Description

This module is intended for use with learners and solves a maximization problem.
As such it expects the negative gradients:

math:: \text{max} f(x)=0

It provides the optimisers

    * fAdaBelief
    * fAdaGrad
    * fAdam
    * fMadGrad
    * fMomentum
    * fRMSProp
    * fSGD

**TODO**: adapt the test suit: :file:`tests/statistical/optimizers/fast/fastOpt_tester.cpp`

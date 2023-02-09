******************************************************************
Discrete CMAES (Covariance Matrix Adaptation Evolution Strategy)
******************************************************************

This is the implementation of the discrete variant of  *Covariance Matrix Adaptation Evolution Strategy*, as published in `Benhamou20019 <https://hal.science/hal-02011531/document>`_.
In contrast to the continuous variant, new candidate solutions are sampled according to a multivariate binomial distribution. All the other properties are inherited from the original version, and we refer to the README for CMAES for details.

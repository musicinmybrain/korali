import numpy as np
from scipy.integrate import odeint

def solveLinearODEs(theta, IC, tEval):
    T0 = 0
    Tend = max(tEval)
    F = lambda y, t, theta: np.array([y[1]*theta[0],-y[0]*theta[1]])
    y = odeint(F, IC, tEval, args=(theta,))
    print(y)
    return np.sum(y, axis=1)



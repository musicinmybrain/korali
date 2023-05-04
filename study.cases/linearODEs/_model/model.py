import numpy as np
from scipy.integrate import odeint

def solveLinearODEs(theta, IC, tEval):
    T0 = 0
    Tend = max(tEval)
    F = lambda y, t, theta: np.array([-theta[0]*y[1],theta[1]*y[0]])
    y = odeint(F, IC, tEval, args=(theta,))
    #return y[:,0] #np.sum(y, axis=1)
    return y[:,0]+y[:,1] #np.sum(y, axis=1)

#!/bin/python3

# Discretization grid
N = 512
N2 = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./_model/')

import numpy as np
from plotting_jax import *
from Burger_jax import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*pi
dt      = 0.001
s       = N/N2
tEnd    = 100
nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
ic      = 'sinus'
#ic      = 'forced'
noise   = 0.
seed    = 42

dns =  Burger_jax(L=L, N=N,  dt=dt,   nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
sgs0 = Burger_jax(L=L, N=N2, dt=dt*s, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)

v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
sgs0.IC( v0 = v0 * N2 / N )

#------------------------------------------------------------------------------
print("Simulate DNS ..")
# simulate
dns.simulate()
dns.compute_Ek()

print("Simulate SGS..")
# simulate
sgs0.simulate()
# convert to physical space
sgs0.compute_Ek()
sgs = sgs0

#------------------------------------------------------------------------------
print(dns.uu[-1])

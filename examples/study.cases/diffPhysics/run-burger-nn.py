# GET REFERENCE SOLUTION AND LOW RESOLUTION SOLUTION TO BE TRAINED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./_model/')

import numpy as np
from plotting import *
from Burger import *

#------------------------------------------------------------------------------
# Initialization

# Discetization grid
N = 1024
N2 = 32

# Set parameters
L       = 2*pi
dt      = 0.001
s       = 32
tEnd    = 100
nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
ic      = 'sinus'
#ic      = 'forced'
noise   = 0.
seed    = 42
forcing = True

# Initialize both versions
dns = Burger(L=L, N=N,  dt=dt,   nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)
sgs = Burger(L=L, N=N2, dt=dt*s, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)

v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
sgs.IC( v0 = v0 * N2 / N )

sgs.randfac1 = dns.randfac1
sgs.randfac2 = dns.randfac2

#------------------------------------------------------------------------------
# Perform simulations for both versions

print("Simulate DNS ..")
## simulate
dns.simulate()
dns.compute_Ek()

print("Simulate SGS..")
## simulate
sgs.simulate()
sgs.compute_Ek()


#------------------------------------------------------------------------------
# Store solutions
dns_sol = dns.uu
sgs_sol = sgs.uu
print(sgs_sol[-1])



# TODO: IMPLEMENT TRAINING FROM neural-network-baseline.py



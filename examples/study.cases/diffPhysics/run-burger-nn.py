# GET REFERENCE SOLUTION AND LOW RESOLUTION SOLUTION TO BE TRAINED

import sys
sys.path.append('./_model/')

import numpy as np
from Burger_jax import *

#------------------------------------------------------------------------------
# Initialization

# Discetization grid (those should be 2^integer)
N = 512 # Discretization / number of grid points of DNS
N2 = 32 # Discretization / number of grid points of UGS

# Set parameters
L       = 2*pi  # Length of domain
dt      = 0.001 # Simulator time step
tEnd    = 5     # Duration simulation / end time
nu      = 0.02  # Viscosity
#ic      = 'zero'
#ic      = 'turbulence'
ic      = 'sinus' # initial condition
#ic      = 'forced'
noise   = 0.    # Standard deviation of IC
seed    = 42    # Random seed
forcing = False # Use forcing term in equation

# Adapt time steps and end time (just some testing)
new_fraction = 1.0
dt   *= new_fraction
tEnd *= new_fraction

# Compute time step for sgs
s       = int(N/N2)
dt_sgs  = dt*s

# Initialize both versions
dns = Burger_jax(L=L, N=N,  dt=dt,     nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
sgs = Burger_jax(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
#dns = Burger(L=L, N=N,  dt=dt,     nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)
#sgs = Burger(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)

v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
sgs.IC( v0 = v0 * N2 / N )

#sgs.randfac1 = dns.randfac1
#sgs.randfac2 = dns.randfac2

#------------------------------------------------------------------------------
# Perform simulations for both versions

print("Simulate DNS ..")
## simulate
dns.simulate()
dns.compute_Ek()

print("Simulate SGS ..")
## simulate
sgs.simulate()
sgs.compute_Ek()

#------------------------------------------------------------------------------
# Store solutions
dns_sol = dns.uu
sgs_sol = sgs.uu

# Prepare variables for training
dns_indices = range(0, N, s)
dns_short_sol = dns_sol[:, dns_indices]
#print(dns_short_sol[-1])
#print(sgs_sol[-1])

# Test: Those should be the same
#print(sgs.x)
#print(dns.x[dns_indices])

# Test: Print some solutions
#number_of_sol = 3
#for i in range(number_of_sol):
#    t = i * tEnd/number_of_sol
#    dns_idx = int(t/dt)
#    sgs_idx = int(t/dt_sgs)
#    print("SGS solutions:")
#    print(sgs_sol[sgs_idx, :])
#    print("DNS solutions:")
#    print(dns_sol[dns_idx, dns_indices])
#    print("\n")

print("Simulations done. Start with the training ..")





# TRAINING WITH NEURAL NETWORK

import jax.numpy as jnp
from jax import random

from jax.example_libraries import optimizers

from NeuralNetworkJax import *

#------------------------------------------------------------------------------
# Initialization

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)

# Define dimensions
batch_dim = 500    # Number of iterations in training step (number of different time steps)
feature_dim = N2   # Size of input / output (size of x-grid)
hidden_dim = 256   # Size / width of hidden layer
batch_size = feature_dim

# Number of training epochs (per run) in training function
num_epochs = 100

# Generate Gaussian weights and biases
params = [random.normal(key, (hidden_dim, feature_dim)),
          random.normal(key, (hidden_dim, ))]

# Define layer, parameters and optimal state
layer_sizes = [feature_dim, hidden_dim, hidden_dim, feature_dim]
# Return a list of tuples of layer weights
params = initialize_weights(layer_sizes, key)
opt_state = opt_init(params)

#------------------------------------------------------------------------------
# Run the training function and get losses and optimal parameters / states
train_loss, params_new, opt_state_new = run_training_loop(num_epochs, opt_state, batch_dim, batch_size, dns_short_sol, sgs_sol, tEnd, dt, dt_sgs)

# Optional: Plot losses (mean or 16 losses for different times)
losses = np.array(train_loss).reshape(num_epochs, batch_dim)
PlotMeanLoss(losses, num_epochs, batch_dim)
PlotLosses(losses, num_epochs, batch_dim)

#------------------------------------------------------------------------------
# Get corrections
corrections, predictions = get_corrections(opt_state_new, batch_size, sgs_sol, tEnd, dt_sgs)

# Simulate a new base solution with same data than sgs solution
print("Simulate new solution ..")
base = Burger_jax(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
base.IC( v0 = v0 * N2 / N )
# Apply correction
base.step(correction = np.array(corrections))
base.ioutnum = 0 # This has to be reset manually before we start a new simulation
base.t = 0.0     # This has to be reset manually before we start a new simulation
base.stepnum = 0 # This has to be reset manually before we start a new simulation
base.simulate()

# Plot solutions (plot predicted value at the end of training)
print("Plotting Solutions and Prediction ..")
test_losses = PlotSolsAndPredict(sgs_sol, dns_short_sol, sgs.x, batch_size, opt_state_new, tEnd, dt, dt_sgs)

# Plot solutions (plot testing values)
from plotting import makePlot
makePlot(dns, base, sgs, "FeedforwardNN")

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
step_noise = 0.05 # Standard deviation of gaussian noise in steps
noise_seed = 42   # Seed for noise in steps

# Compute time step for sgs
s       = int(N/N2)
dt_sgs  = dt*s

# Initialize both versions
dns = Burger_jax(L=L, N=N,  dt=dt,     nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
sgs = Burger_jax(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)

v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
sgs.IC( v0 = v0 * N2 / N )

#------------------------------------------------------------------------------
# Perform simulations for both versions

print("Simulate DNS ..")
## simulate
dns.simulate()

print("Simulate SGS ..")
## simulate
sgs.simulate()

#------------------------------------------------------------------------------
# Store solutions
dns_sol = dns.uu
sgs_sol = sgs.uu

# Prepare variables for training
dns_indices = range(0, N, s)
dns_short_sol = dns_sol[:, dns_indices]

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
num_epochs = 300 # 300 gives decent results and does not take too long

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
train_loss, params_new, opt_state_new = run_training_loop(num_epochs, opt_state, batch_dim, batch_size, dns_short_sol, sgs_sol, tEnd, dt, dt_sgs, step_noise, noise_seed)

# Plot solutions (plot predicted value at the end of training)
print("Plotting Solutions and Prediction ..")
PlotSolsAndPredict(sgs_sol, dns_short_sol, sgs.x, batch_size, opt_state_new, tEnd, dt, dt_sgs)

# Optional: Plot losses (mean or 16 losses for different times)
losses = np.array(train_loss).reshape(num_epochs, batch_dim)
PlotMeanLoss(losses, num_epochs, batch_dim)
PlotLosses(losses, num_epochs, batch_dim)

#------------------------------------------------------------------------------
# Simulate a new base solution with same data than sgs solution
print("Simulate new solution ..")
base = Burger_jax(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
base.IC( v0 = v0 * N2 / N )

# Apply corrections
# advance in time for nsteps steps
try:
    for n in range(1,base.nsteps+1):
        #correction, _ = get_corrections(opt_state_new, batch_size, base.uu, base.ioutnum)
        correction, _ = get_corrections(opt_state_new, batch_size, base.uu[n])
        base.step(correction = np.array(correction))

except FloatingPointError:
    print("[Burger_jax] Floating point exception occured", flush=True)
    # something exploded
    # cut time series to last saved solution and return
    base.nout = base.ioutnum
    base.vv.resize((base.nout+1,base.N)) # nout+1 because the IC is in [0]
    base.tt.resize(base.nout+1)          # nout+1 because the IC is in [0]

# Plot solutions (plot testing values)
from plotting import makePlot
makePlot(dns, base, sgs, "FeedforwardNN")

## Test a new mu
#dns2  = Burger_jax(L=L, N=N,  dt=dt,     nu=nu*0.9, tend=tEnd, case=ic, noise=noise, seed=seed)
#base2 = Burger_jax(L=L, N=N2, dt=dt_sgs, nu=nu*0.9, tend=tEnd, case=ic, noise=noise, seed=seed)
#v0 = np.concatenate((dns2.v0[:((N2+1)//2)], dns2.v0[-(N2-1)//2:]))
#base2.IC( v0 = v0 * N2 / N )
#print("Simulate DNS ..")
### simulate
#dns2.simulate()
#dns2.compute_Ek()
#print("Simulate SGS ..")
### simulate
#base2.simulate()
#base2.compute_Ek()
#dns2_sol = dns2.uu
#base2_sol = base2.uu
#dns2_short_sol = dns2_sol[:, dns_indices]
#print("Plotting Solutions and Prediction ..")
#PlotSolsAndPredict(base2_sol, dns2_short_sol, base2.x, batch_size, opt_state_new, tEnd, dt, dt_sgs)


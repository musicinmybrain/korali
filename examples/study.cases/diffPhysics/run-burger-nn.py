# Tuning parameters
step_noise = 0.00 # Standard deviation of gaussian noise in steps (use 0 for no noise)
noise_seed = 42   # Seed for noise in steps (change it to get slightly different results)
levels     = 1    # Levels of propagation learning (use 1 for default)





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
pre_epochs = 0   # Learning before the correction if levels > 0

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
# Note: This is only fully done if there is one level only
if levels == 1:
    train_loss, params_new, opt_state_new = run_training_loop(num_epochs, opt_state, batch_dim, batch_size, dns_short_sol, sgs_sol, tEnd, dt, dt_sgs, step_noise, noise_seed)

else:
    train_loss, params_new, opt_state_new = run_training_loop(pre_epochs, opt_state, batch_dim, batch_size, dns_short_sol, sgs_sol, tEnd, dt, dt_sgs, step_noise, noise_seed)

# Store losses
total_loss = train_loss

# Initialize array for corrections
corrections = []

#------------------------------------------------------------------------------
# Run one or more testings (and learn testing if levels > 1)
for level in range(1, levels+1):
    # Simulate a new base solution with same data than sgs solution
    print(f"Simulate new solution ({level} / {levels}) ..")
    base = Burger_jax(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
    base.IC( v0 = v0 * N2 / N )

    # Apply corrections / advance in time for nsteps steps
    try:
        for n in range(1,base.nsteps+1):
            # Save corrections for a plot if on final level
            if level == levels:
                corrected = base.step(correction = True, opt_state = opt_state_new)
                corrections.append(corrected)
            # Proceed as normal if not on final level
            else:
                base.step(correction = True, opt_state = opt_state_new)

    except FloatingPointError:
        print("Floating point exception occured", flush=True)
        # something exploded
        # cut time series to last saved solution and return
        #base.nout = base.ioutnum
        #base.vv.resize((base.nout+1,base.N)) # nout+1 because the IC is in [0]
        #base.tt.resize(base.nout+1)          # nout+1 because the IC is in [0]

    # Train on corrections if not already in the final level
    if level < levels:
        # Get array to be trained
        train_arr, tEnd_idx = get_train_arr(base.uu, sgs_sol, int(tEnd/dt_sgs)+1)
        tEnd_new = tEnd_idx * dt_sgs
        print(f"Train new solution until t = {tEnd_new}")

        # Run the training function and update losses and parameters / optimal state
        #train_loss, params_new, opt_state_new = run_training_loop(num_epochs, opt_state_new, batch_dim, batch_size, dns_short_sol, train_arr, tEnd, dt, dt_sgs, step_noise, noise_seed+level)
        train_loss, params_new, opt_state_new = run_training_loop(num_epochs, opt_state_new, batch_dim, batch_size, dns_short_sol, train_arr, tEnd_new, dt, dt_sgs, step_noise, noise_seed+level)

        # Store losses
        total_loss = np.concatenate((total_loss, train_loss), axis=0)

#------------------------------------------------------------------------------
# Plot solutions (plot predicted value at the end of training)
print("Plotting Solutions and Prediction ..")
PlotSolsAndPredict(sgs_sol, dns_short_sol, sgs.x, batch_size, opt_state_new, tEnd, dt, dt_sgs)

# Optional: Plot animated solutions
print("Plotting animated Solutions and Prediction ..")
SolsAndPredictAnimation(sgs_sol, dns_short_sol, sgs.x, batch_size, opt_state_new, tEnd, dt, dt_sgs)

# Optional: Plot losses (mean or 16 losses for different times)
print("Plotting Losses ..")
if levels == 1:
    loss_dim = num_epochs
else:
    loss_dim = pre_epochs + num_epochs * (levels-1)
losses = np.array(total_loss).reshape(loss_dim, batch_dim)
PlotMeanLoss(losses, loss_dim, batch_dim)
PlotLosses(losses, loss_dim, batch_dim)

# Plot solutions (plot testing values)
print("Plotting Testing Solution ..")
PlotTesting(dns, base, sgs, "FeedforwardNN")
# Only first 16 time steps
PlotTesting(dns, base, sgs, "FeedforwardNN_short", False)

# Optional: Plot animated solutions
print("Plotting animated Testing Solution ..")
TestingAnimation(dns, base, sgs, "FeedforwardNN_Animation")

# Optional: Plot corrections
print("Plotting corrections ..")
PlotCorrections(base.uu, corrections, sgs.x, tEnd, dt_sgs, "FeedforwardNN")
# Only first 16 time steps
PlotCorrections(base.uu, corrections, sgs.x, tEnd, dt_sgs, "FeedforwardNN_short", full = False)


# GET REFERENCE SOLUTION AND LOW RESOLUTION SOLUTION TO BE TRAINED

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import sys
#import argparse
sys.path.append('./_model/')

import numpy as np
#from plotting import *
from Burger import *

#------------------------------------------------------------------------------
# Initialization

# Discetization grid (those should be 2^integer)
N = 512
N2 = 32

# Set parameters
L       = 2*pi
dt      = 0.001
tEnd    = 100
nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
ic      = 'sinus'
#ic      = 'forced'
noise   = 0.
seed    = 42
forcing = True

s       = int(N/N2)
dt_sgs  = dt*s

# Initialize both versions
dns = Burger(L=L, N=N,  dt=dt,     nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)
sgs = Burger(L=L, N=N2, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)

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
from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax.example_libraries import optimizers

#import torch
#from torchvision import datasets, transforms

import time

#------------------------------------------------------------------------------
# Initialization and define functions

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)

# Define ReLU function
def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)

jit_ReLU = jit(ReLU)

# Define dimensions
batch_dim = 200   # Number of iterations in training step (number of different time steps)
feature_dim = N2  # Size of input / output (size of x-grid)
hidden_dim = 1024  # Hidden layers in network
batch_size = feature_dim

# Defining an optimizer in Jax
step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)

# Number of training epochs in in training function
num_epochs = 10

# Generate a batch of vectors to process
X = random.normal(key, (batch_dim, feature_dim))

# Generate Gaussian weights and biases
params = [random.normal(key, (hidden_dim, feature_dim)),
          random.normal(key, (hidden_dim, ))]

# Define ReLu layers
def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(jnp.dot(params[0], x) + params[1])

def vmap_relu_layer(params, x):
    """ vmap version of the ReLU layer """
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))

# Initialize weights
def initialize_mlp(sizes, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Define layer, parameters and optimal state
layer_sizes = [feature_dim, hidden_dim, hidden_dim, feature_dim]
# Return a list of tuples of layer weights
params = initialize_mlp(layer_sizes, key)
opt_state = opt_init(params)

# Forward pass / predict function
def forward_pass(params, in_array):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:-1]:
        activations = relu_layer([w, b], activations)

    # Perform final trafo to logits
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits

# Make a batched version of the `predict` function
batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)

# MSE loss
def loss(params, in_arrays, targets):
    """ Compute the mean squared error loss """
    preds = batch_forward(params, in_arrays)
    return ((preds-targets)**2).mean()

# Update function
@jit
def update(params, x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

#------------------------------------------------------------------------------
# Training function
def run_mnist_training_loop(num_epochs, opt_state, net_type="MLP"):
    """ Implements a learning loop over epochs. """
    # Initialize placeholder for loggin
    train_loss = []

    # Get the initial set of parameters
    params = get_params(opt_state)

    # Loop over the training epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(batch_dim):
            t = i * tEnd/batch_dim
            dns_idx = int(t/dt)
            sgs_idx = int(t/dt_sgs)
            dns_arr = dns_short_sol[dns_idx]
            sgs_arr = sgs_sol[sgs_idx]
            x_arr = sgs_arr
            y_arr = dns_arr
            x = jnp.array(x_arr).reshape(1, batch_size)
            y = jnp.array(y_arr[:, None] == jnp.arange(1), jnp.float32)
            params, opt_state, loss = update(params, x, y, opt_state)
            train_loss.append(loss)

        latest_loss = train_loss[-1]
        epoch_time = time.time() - start_time
        print("Epoch {} | T: {:0.2f} | Latest Loss: {:0.10f}".format(epoch+1, epoch_time, latest_loss))

    return train_loss

# Run the function
train_loss = run_mnist_training_loop(num_epochs, opt_state, net_type="MLP")

#------------------------------------------------------------------------------
# Testing
sgs_final = sgs_sol[-1]
dns_final = dns_short_sol[-1]
x = jnp.array(sgs_final).reshape(1, batch_size)
test = vmap_relu_layer(params, x)
print(dns_final)
print(test)

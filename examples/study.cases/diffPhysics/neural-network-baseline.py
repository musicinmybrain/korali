# This script is an adpated version of:
# https://roberttlange.github.io/posts/2020/03/blog-post-10/

import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random

import math

from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers

import torch
from torchvision import datasets, transforms

import time
#from helpers import plot_mnist_examples

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)

# Define ReLU function
def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)

jit_ReLU = jit(ReLU)

# Define dimensions
batch_dim = 128
feature_dim = 64
hidden_dim = 512

# Generate a batch of vectors to process
X = random.normal(key, (batch_dim, feature_dim))

# Generate Gaussian weights and biases
params = [random.normal(key, (hidden_dim, feature_dim)),
          random.normal(key, (hidden_dim, ))]

def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(np.dot(params[0], x) + params[1])

def vmap_relu_layer(params, x):
    """ vmap version of the ReLU layer """
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))

batch_size = feature_dim

# Very basic way to create training data
def create_data(batch_idx):
    x_array = np.zeros(batch_size)
    y_array = np.zeros(batch_size)
    for i in range(batch_size):
        x_array = x_array.at[i].set(batch_idx + i/100.0)
        y_array = y_array.at[i].set(math.sin(x_array[i]))
    return x_array, y_array

# Initialize weights
def initialize_mlp(sizes, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [feature_dim, hidden_dim, hidden_dim, feature_dim]
# Return a list of tuples of layer weights
params = initialize_mlp(layer_sizes, key)

# Forward pass / predict function
def forward_pass(params, in_array):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:-1]:
        activations = relu_layer([w, b], activations)

    # Perform final trafo to logits
    final_w, final_b = params[-1]
    logits = np.dot(final_w, activations) + final_b
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

# Defining an optimizer in Jax
step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

num_epochs = 10

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
            x_arr, y_arr = create_data(i)
            x = np.array(x_arr).reshape(1, batch_size)
            y = np.array(y_arr[:, None] == np.arange(1), np.float32)
            params, opt_state, loss = update(params, x, y, opt_state)
            train_loss.append(loss)
            # print(loss) # This prints the loss after every single step

        latest_loss = train_loss[-1]
        epoch_time = time.time() - start_time
        print("Epoch {} | T: {:0.2f} | Latest Loss: {:0.10f}".format(epoch+1, epoch_time, latest_loss))

    return train_loss

train_loss = run_mnist_training_loop(num_epochs, opt_state, net_type="MLP")

import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax.example_libraries import optimizers

import time



# Define ReLU function
def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)

# Define tanh non-linearity function
def tanhU(x):
    """ tanh activation function """
    return jnp.tanh(x)

# jit versions of functions above
jit_ReLU  = jit(ReLU)
jit_tanhU = jit(tanhU)


# Defining an optimizer in Jax
step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)


# Define ReLu layers
def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(jnp.dot(params[0], x) + params[1])

# Define tanh layers
def tanh_layer(params, x):
    """ Simple tanh layer for single sample """
    return tanhU(jnp.dot(params[0], x) + params[1])

# Function to initialize weights
def initialize_weights(sizes, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Forward pass / predict function
def forward_pass(params, in_array):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU / tanh hidden layers
    for w, b in params[:-1]:
#        activations = relu_layer([w, b], activations)
        activations = tanh_layer([w, b], activations)

    # Perform final trafo to returned prediction
    final_w, final_b = params[-1]
    prediction = jnp.dot(final_w, activations) + final_b
    return prediction

# Make a batched version of the `predict` function
batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)

# MSE loss
def loss(params, in_arrays, targets):
    """ Compute the mean squared error loss """
    preds = batch_forward(params, in_arrays)
    return jnp.mean((preds - targets)**2)

# Update function
@jit
def update(params, x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value



#------------------------------------------------------------------------------
# Training function
def run_training_loop(num_epochs, opt_state, batch_dim, batch_size, dns, sgs, tEnd, dt_dns, dt_sgs):
    """ Implements a learning loop over epochs. """
    # Initialize placeholder for loggin
    train_loss = []

    # Get the initial set of parameters
    params = get_params(opt_state)

    # Loop over the training epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(batch_dim):
            # Define indices for sgs and dns
            t = i * tEnd/batch_dim
            dns_idx = int(t/dt_dns)
            sgs_idx = int(t/dt_sgs)
            # Prepare variables
            dns_arr = dns[dns_idx]
            sgs_arr = sgs[sgs_idx]
            x = jnp.array(sgs_arr).reshape(1, batch_size)
            y = jnp.array(dns_arr).reshape(1, batch_size)
            # Training step
            params, opt_state, loss = update(params, x, y, opt_state)
            train_loss.append(loss)

        latest_loss = train_loss[-1]
        epoch_time = time.time() - start_time
        print("Epoch {} | T: {:0.2f} | Latest Loss: {:0.10f}".format(epoch+1, epoch_time, latest_loss))

    return train_loss, params, opt_state



#------------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt 

# Plotting function
def PlotSolsAndPredict(sgs, dns, x_arr, batch_size, opt_state, tEnd, dt_dns, dt_sgs):
    # Get optimal parameters
    params = get_params(opt_state)
    losses = []

    # Prepare plot
    figName = "Solution_and_Prediction_Plot.pdf"
    colors = ['black','royalblue','seagreen']
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))

    for i in range(16):
        # Prepare index variables
        t = i * tEnd / 16
        tidx_dns = int(t/dt_dns)
        tidx_sgs = int(t/dt_sgs)
        k = int(i / 4)
        l = i % 4

        # Prepare variables to be plotted
        sgs_sol = sgs[tidx_sgs]
        dns_sol = dns[tidx_dns]
        x = jnp.array(sgs_sol).reshape(1, batch_size)
        y = jnp.array(dns_sol).reshape(1, batch_size)
        params = get_params(opt_state)
        predict = batch_forward(params, x)
        losses.append(loss(params, x, y))

        # Plot and save figure
        axs[k,l].plot(x_arr, sgs_sol,     '-',  color=colors[1], label='SGS solution')
        axs[k,l].plot(x_arr, dns_sol,     '-',  color=colors[0], label='DNS solution')
        axs[k,l].plot(x_arr, predict[-1], '--', color=colors[2], label='Predicted values')

        # Add title, labels, legend and save figure
        #axs[k,l].set_title('Non-Korali version')
        axs[k,l].set_xlabel('x')
        axs[k,l].set_ylabel('u(x)')
        #axs[k,l].legend()

    # Save figure
    fig.savefig(figName)
    print("Plot has been saved under the name: Solution_and_Prediction_Plot.pdf")

#    # Output in text form
#    print("DNS solution:")
#    print(dns_final)
#    print("SGS solution:")
#    print(sgs_final)
#    print("Predictions:")
#    print(predict[-1])
#    print("Loss:")
#    print(loss_final)

    return losses

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
def loss(params, in_arrays, targets, step_noise, key):
    """ Compute the mean squared error loss """
    # Add noise to training
    noise = random.normal(key, (1, 32)) * step_noise

    preds = batch_forward(params, in_arrays+noise)
    return jnp.mean((preds - targets)**2)

# Update function
@jit
def update(params, x, y, opt_state, step_noise, key):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, x, y, step_noise, key)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value



#------------------------------------------------------------------------------
# Training function
def run_training_loop(num_epochs, opt_state, batch_dim, batch_size, dns, sgs, tEnd, dt_dns, dt_sgs, step_noise, noise_seed):
    """ Implements a learning loop over epochs. """
    # Initialize placeholder for losses
    train_loss = []

    # Generate key
    key = random.PRNGKey(noise_seed)

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
            # Create new key
            key, subkey = random.split(key)
            # Training step
            params, opt_state, loss = update(params, x, y, opt_state, step_noise, subkey)
            train_loss.append(loss)

        latest_loss = train_loss[-1]
        epoch_time = time.time() - start_time
        print("Epoch {} | T: {:0.2f} | Latest Loss: {:0.10f}".format(epoch+1, epoch_time, latest_loss))

    return train_loss, params, opt_state



#------------------------------------------------------------------------------
# Function to get corrections
def get_corrections(opt_state, batch_size, sgs):
    # Get the current set of parameters
    params = get_params(opt_state)

    # Prepare variables
    x = jnp.array(sgs).reshape(1, batch_size)

    # Getting corrections
    predict = batch_forward(params, x)
    correction = predict[-1] - sgs

    return correction, predict


# Function to create a training array when base solution explodes
def get_train_arr(base, sgs, t_dim):
    # Maximum absolute value that is tolerated
    max_cap = 2.0

    # Initialize values
    abs_max = 0.0
    i = 0

    # Find maximum index
    while abs_max < max_cap:
        abs_max = jnp.max(jnp.absolute(base[i]))
        i += 1

    # Prepare variables
    base_range = range(i)
    sgs_range = range(i, t_dim)

    # Training array that is returned (meaningful base solutions + rest is filled with SGS solutions)
    train_arr = jnp.concatenate((base[base_range], sgs[sgs_range]), axis=0)
    return train_arr, i

#------------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np

# Plotting function for SGS, DNS and predicted solution
def PlotSolsAndPredict(sgs, dns, x_arr, batch_size, opt_state, tEnd, dt_dns, dt_sgs):
    # Get optimal parameters
    params = get_params(opt_state)

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
        predict = batch_forward(params, x)

        # Plot figures
        axs[k,l].plot(x_arr, sgs_sol,     '-',  color=colors[1], label='SGS solution')
        axs[k,l].plot(x_arr, dns_sol,     '-',  color=colors[0], label='DNS solution')
        axs[k,l].plot(x_arr, predict[-1], '--', color=colors[2], label='Predicted values')

        # Add labels
        #axs[k,l].set_title('Non-Korali version')
        axs[k,l].set_xlabel('x')
        axs[k,l].set_ylabel('u(x)')
        #axs[k,l].legend()

    # Add legend to first plot
    axs[0,0].legend()

    # Fix y range
    plt.ylim([-1.3, 1.3])

    # Save figure
    fig.savefig(figName)
    print("Plot has been saved under the name: Solution_and_Prediction_Plot.pdf")


# Plotting function for SGS, DNS and tested base solution
# Note: This function is copied from plotting.py and has been slightly adapted
def PlotTesting(dns, base, sgs, fileName):
    # Prepare plot
    figName = fileName + "_evolution.pdf"
    colors = ['black','royalblue','seagreen']
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))    

    # Get variables
    tEnd = dns.tend
    dt = dns.dt
    sgs_dt = sgs.dt

    for i in range(16):
        # Prepare index variables
        t = i * tEnd / 16
        tidx = int(t/dt)
        tidx_sgs = int(t/sgs_dt)
        k = int(i / 4)
        l = i % 4
        
        # Plot figures 
        axs[k,l].plot(sgs.x,  sgs.uu[tidx_sgs,:],  '-',  color=colors[1], label='SGS solution')
        axs[k,l].plot(dns.x,  dns.uu[tidx,:],      '-',  color=colors[0], label='DNS solution')
        axs[k,l].plot(base.x, base.uu[tidx_sgs,:], '--', color=colors[2], label='Base solution')

        # Add labels
        axs[k,l].set_xlabel('x')
        axs[k,l].set_ylabel('u(x)')

    # Add legend to first plot
    axs[0,0].legend() 

    # Fix y range
    plt.ylim([-1.3, 1.3])

    # Save Figure
    fig.savefig(figName)
    print(f"Plot has been saved under the name: {figName} ...")


# Plotting function for losses
def PlotLosses(losses, epochs, batch_dim):
    # Prepare plot
    figName = "Loss_Plot.pdf"
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))

    # If there are too many epochs, only plot some of them
    if epochs > 100:
        step_size = int(epochs / 100)
        plot_indices = range(0, epochs, step_size)
        # x-range
        x_arr = range(1, epochs+1, step_size)
    else:
        plot_indices = range(0, epochs, 1)
        # x-range
        x_arr = range(1, epochs+1, 1)

    for i in range(16):
        # Prepare index variables
        epoch_idx= int(i*batch_dim / 16)
        k = int(i / 4)
        l = i % 4

        # Prepare variables to be plotted
        loss_arr = losses[:, epoch_idx]

        # Plot figures
        axs[k,l].plot(x_arr, loss_arr[plot_indices], '-', label='Loss')
        axs[k,l].set_yscale('log')

        # Add labels
        axs[k,l].set_xlabel('Training epoch')
        axs[k,l].set_ylabel('Loss')

    # Save figure
    fig.savefig(figName)
    print("Plot has been saved under the name: Loss_Plot.pdf")

# Plotting function for mean loss
def PlotMeanLoss(losses, epochs, batch_dim):
    # Prepare plot
    figName = "MeanLoss_Plot.pdf"
    fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(15,15))
    
    if epochs > 100:
        step_size = int(epochs / 100)
        plot_indices = range(0, epochs, step_size)
        # x-range
        x_arr = range(1, epochs+1, step_size)
    else:
        plot_indices = range(0, epochs, 1)
        # x-range
        x_arr = range(1, epochs+1, 1)

    # Prepare variables to be plotted
    loss_arr = np.mean(losses, axis=1)

    # Plot figures
    axs.plot(x_arr, loss_arr[plot_indices], '-', label='Mean Loss')
    axs.set_yscale('log')

    # Add labels
    axs.set_xlabel('Training epoch')
    axs.set_ylabel('Mean Loss')

    # Save figure
    fig.savefig(figName)
    print("Plot has been saved under the name: MeanLoss_Plot.pdf")

#------------------------------------------------------------------------------
import matplotlib.animation as animation

# Plotting animated function for SGS, DNS and predicted solution
def SolsAndPredictAnimation(sgs, dns, x_arr, batch_size, opt_state, tEnd, dt_dns, dt_sgs):
    # Get optimal parameters
    params = get_params(opt_state)

    # Compute frames and ratio dt_sgs / dt_dns
    frames = int(tEnd/dt_sgs)
    s = int(dt_sgs/dt_dns)

    # Prepare plot
    fig = plt.figure()
    colors = ['black','royalblue','seagreen']
    axis = plt.axes(xlim = (0, 6.3), ylim = (-1.3, 1.3))

    line1, = axis.plot([], [], '-',  color=colors[1], label='SGS solution')
    line2, = axis.plot([], [], '-',  color=colors[0], label='DNS solution')
    line3, = axis.plot([], [], '--', color=colors[2], label='Predicted values')

    # Initializing function
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3,

    # Animation function
    def animate(i):
        # Prepare variables to be plotted
        sgs_sol = sgs[i]
        dns_sol = dns[i*s]
        x = jnp.array(sgs_sol).reshape(1, batch_size)
        y = jnp.array(dns_sol).reshape(1, batch_size)
        predict = batch_forward(params, x)

        # Assign values
        line1.set_data(x_arr, sgs_sol)
        line2.set_data(x_arr, dns_sol)
        line3.set_data(x_arr, predict[-1])

        return line1, line2, line3,

    # Call the animation function    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = frames, interval = 20, blit = True)

    # Add labels
    axis.set_xlabel('x')
    axis.set_ylabel('u(x)')

    # Save animation
    anim.save('Solution_and_Prediction_Animation.mp4', writer = 'ffmpeg', fps = 30)
    print("Animation has been saved under the name: Solution_and_Prediction_Animation.mp4")


# Plotting animated function for SGS, DNS and tested base solution
def TestingAnimation(dns, base, sgs, fileName):
    # Get variables
    tEnd = dns.tend
    dt = dns.dt
    sgs_dt = sgs.dt

    # Compute frames and ratio dt_sgs / dt_dns
    frames = int(tEnd/sgs_dt)
    s = int(sgs_dt/dt)

    # Prepare plot
    figName = fileName + "_evolution.mp4"
    fig = plt.figure()
    colors = ['black','royalblue','seagreen']
    axis = plt.axes(xlim = (0, 6.3), ylim = (-1.3, 1.3))

    line1, = axis.plot([], [], '-',  color=colors[1], label='SGS solution')
    line2, = axis.plot([], [], '-',  color=colors[0], label='DNS solution')
    line3, = axis.plot([], [], '--', color=colors[2], label='Base solution')

    # Initializing function
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3,

    # Animation function
    def animate(i):
        # Prepare variables to be plotted
        sgs_sol  = sgs.uu[i]
        dns_sol  = dns.uu[i*s]
        base_sol = base.uu[i]

        # Assign values
        line1.set_data(sgs.x,  sgs_sol)
        line2.set_data(dns.x,  dns_sol)
        line3.set_data(base.x, base_sol)

        return line1, line2, line3,

    # Call the animation function    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = frames, interval = 20, blit = True)

    # Add labels
    axis.set_xlabel('x')
    axis.set_ylabel('u(x)')

    # Save animation
    anim.save(figName, writer = 'ffmpeg', fps = 10)
    print(f"Animation has been saved under the name: {figName} ...")


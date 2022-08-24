# This script is based on the following source code:
# https://www.physicsbaseddeeplearning.org/overview-burgers-forw.html

# Use the latest PhiFlow version for this
# !pip install --upgrade --quiet phiflow



# SIMULATION

from phi.flow import *

# define variables
N = 128             # cells / discretization points
DX = 2./N           # length of cell (interval is [-1, 1] by defaultl)
STEPS = 32          # time steps 
DT = 1./STEPS       # length of time step (time interval is 1 by default)
NU = 0.01/(N*np.pi) # viscosity

# initialization of velocities, cell centers of a CenteredGrid have DX/2 offsets for linspace()
INITIAL_NUMPY = np.asarray( [-np.sin(np.pi * x) for x in np.linspace(-1+DX/2,1-DX/2,N)] ) # 1D numpy array
INITIAL = math.tensor(INITIAL_NUMPY, spatial('x') ) # convert to phiflow tensor

# define velocity object
velocity = CenteredGrid(INITIAL, extrapolation.PERIODIC, x=N, bounds=Box[-1:1])
# vt = advect.semi_lagrangian(velocity, velocity, DT)

# alternative velocity objects
#velocity = CenteredGrid(lambda x: -math.sin(np.pi * x), extrapolation.PERIODIC, x=N, bounds=Box[-1:1])
#velocity = CenteredGrid(Noise(), extrapolation.PERIODIC, x=N, bounds=Box[-1:1]) # random init

# Use this to print the velocity objects
#print("Velocity tensor shape: "   + format( velocity.shape )) # == velocity.values.shape
#print("Velocity tensor type: "    + format( type(velocity.values) ))
#print("Velocity tensor entries 10 to 14: " + format( velocity.values.numpy('x')[10:15] ))

# running the simulation
velocities = [velocity]
age = 0.
for i in range(STEPS):
    v1 = diffuse.explicit(velocities[-1], NU, DT)
    v2 = advect.semi_lagrangian(v1, v1, DT)
    age += DT
    velocities.append(v2)

print("New velocity content at t={}: {}".format( age, velocities[-1].values.numpy('x,vector')[0:5] ))



# VISUALIZATION 

# get "velocity.values" from each phiflow state with a channel dimensions, i.e. "vector"
vels = [v.values.numpy('x,vector') for v in velocities] # gives a list of 2D arrays

import pylab

fig = pylab.figure().gca()
fig.plot(np.linspace(-1,1,len(vels[ 0].flatten())), vels[ 0].flatten(), lw=2, color='blue',  label="t=0")
fig.plot(np.linspace(-1,1,len(vels[10].flatten())), vels[10].flatten(), lw=2, color='green', label="t=0.3125")
fig.plot(np.linspace(-1,1,len(vels[20].flatten())), vels[20].flatten(), lw=2, color='cyan',  label="t=0.625")
fig.plot(np.linspace(-1,1,len(vels[32].flatten())), vels[32].flatten(), lw=2, color='purple',label="t=1")
pylab.xlabel('x'); pylab.ylabel('u'); pylab.legend()

def show_state(a, title):
    # we only have 33 time steps, blow up by a factor of 2^4 to make it easier to see
    # (could also be done with more evaluations of network)
    a=np.expand_dims(a, axis=2)
    for i in range(4):
        a = np.concatenate( [a,a] , axis=2)

    a = np.reshape( a, [a.shape[0],a.shape[1]*a.shape[2]] )
    #print("Resulting image size" +format(a.shape))

    fig, axes = pylab.subplots(1, 1, figsize=(16, 5))
    im = axes.imshow(a, origin='upper', cmap='inferno')
    pylab.colorbar(im) ; pylab.xlabel('time'); pylab.ylabel('x'); pylab.title(title)
        


# SAVE DATA

vels_img = np.asarray( np.concatenate(vels, axis=-1), dtype=np.float32 )

# save for comparison with reconstructions later on
import os; os.makedirs("./temp",exist_ok=True)
np.savez_compressed("./temp/burgers-groundtruth-solution.npz", np.reshape(vels_img,[N,STEPS+1])) # remove batch & channel dimension

show_state(vels_img, "Velocity")



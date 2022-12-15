import os
import numpy as np
import matplotlib.pyplot as plt

from helpers import average_profile

name = 'slowdiff/'
folder='/scratch/snx3000/anoca/CUP2D/' + name
files = ['x_velocity_profile_0.dat', 'y_velocity_profile_0.dat']
data_folder = '../data/'
plot_folder = '../plots/'

# os.mkdir(data_folder)

# x-direction
x_vel = np.genfromtxt(folder + files[0], delimiter=' ', skip_header=1)

# plot an arbitrary component of the velocity profile
plt.figure(1)
plt.plot(x_vel[:, 0], x_vel[:, 5])
plt.savefig(plot_folder + 'random_profile_x.png')

plt.figure(2)
res_x = average_profile(x_vel, 720, 1200)

plt.plot(res_x)
plt.savefig(plot_folder + 'averaged_profile_x.png')

np.savetxt( data_folder + 'x_profile.dat', res_x[[-1], :])


# y-direction
y_vel = np.genfromtxt(folder + files[1], delimiter=' ', skip_header=1)

plt.figure(3)
plt.plot(y_vel[:, 0], y_vel[:, 5])
plt.savefig(plot_folder + 'random_profile_y.png')

plt.figure(4)
res_y = average_profile(y_vel, 720, 1200)

plt.plot(res_y)
plt.savefig(plot_folder + 'averaged_profile_y.png')

np.savetxt(data_folder + 'y_profile.dat', res_y[[-1], :])
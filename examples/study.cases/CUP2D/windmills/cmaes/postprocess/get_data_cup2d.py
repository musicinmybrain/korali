import os
import numpy as np

from helpers import average_profile

folder='/scratch/snx3000/anoca/CUP2D/cmaes/'

num_sims = 10
names = [folder + f"{i}/" for i in range(1, 1 + num_sims)]

files = ['x_velocity_profile_0.dat', 'y_velocity_profile_0.dat']
output_folder = '../data/'

x_profiles = np.zeros((num_sims, 16))
y_profiles = np.zeros((num_sims, 16))

for ind, fol in enumerate(names):
    x_vel = np.genfromtxt(fol + files[0], delimiter=' ', skip_header=0)
    res_x = average_profile(x_vel, 700, 1200)
    x_profiles[ind] = res_x[-1, :]

    y_vel = np.genfromtxt(fol + files[1], delimiter=' ', skip_header=0)
    res_y = average_profile(y_vel, 700, 1200)
    y_profiles[ind] = res_y[-1, :]

np.savez(output_folder + 'cup2d_data.npz', x_profiles=x_profiles, y_profiles=y_profiles)
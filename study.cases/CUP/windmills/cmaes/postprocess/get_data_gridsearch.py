import matplotlib.pyplot as plt
import numpy as np
import os
from helpers import average_profile

# this file is used to convert all the velocity profiles of each simulation into a time-averaged profile
# it also allows to compute the ratio between the simulations

correct=[3, -3, 0.25, 0.5]
# constant is the a1 = 3 f1 = 0.25
# the varying fan had a solution of a2 = -3 f = 0.5
# varying fan between a = -3.5 to 3.5 and f between 0 and 5, 71 and 51
def getRatiosFromName(name):
    # name has format A*.*_f*.*
    cleaned_name = name.replace("A", "").replace("f", "")
    ind_ = cleaned_name.index('_')
    # print(cleaned_name)
    amplitude = float(cleaned_name[:ind_])
    frequency = float(cleaned_name[ind_+1:])
    # return amplitude, frequency
    r_a = amplitude / correct[0]
    r_f = frequency / correct[2]
    return r_a, r_f

folder = '/scratch/snx3000/anoca/CUP2D/gridsearch/'

subfolders_name = [os.path.basename(f.path) for f in os.scandir(folder) if f.is_dir()]

num_sims_a = 71
num_sims_f = 66

def sortFolders(name):
    # given name of folder, returns a number that allows to order the folders
    r_a, r_f = getRatiosFromName(name)
    return (r_a, r_f)


# print(sorted(subfolders_name, key=sortFolders))
count = 0
sorted_subfolders_name = []
temp_folder = []
#####
for f in sorted(subfolders_name, key=sortFolders):
    temp_folder.append(f)
    count += 1

    if count >= num_sims_f:
        sorted_subfolders_name.append(temp_folder)
        temp_folder = []
        # sorted_subfolders_name[count1] = sorted(sorted_subfolders_name[count1])
        count = 0
        

# sorted_subfolders_name.pop()
print(sorted_subfolders_name[-1])

##### save the ratios file

# size is : 71 x 66 x 2
ratios = np.zeros((num_sims_a, num_sims_f, 2))
for index, subf in enumerate(sorted_subfolders_name):
    for index2, subf2 in enumerate(subf):
        r_a, r_f = getRatiosFromName(subf2)
        ratios[index, index2, 0] = r_a
        ratios[index, index2, 1] = r_f

# save the ratios to a numpy file
np.save('../data/ratios.npy', ratios)

##### save the data file

# size is num_sims_a x num_sims_f x dir_vel x vel_profile_pts

data = np.zeros((num_sims_a, num_sims_f, 2, 16))

for index, list_folder in enumerate(sorted_subfolders_name):
    print(index)
    for index2, folder_name in enumerate(list_folder):

        data_x = np.genfromtxt(folder + folder_name + '/x_velocity_profile_0.dat', delimiter=' ', skip_header=1)
        data_y = np.genfromtxt(folder + folder_name + '/y_velocity_profile_0.dat', delimiter=' ', skip_header=1)

        res_x = average_profile(data_x, 720, 1200)
        res_y = average_profile(data_y, 720, 1200)

        data[index, index2, 0, :] = res_x[-1]
        data[index, index2, 1, :] = res_y[-1]

np.save('../data/final_profiles.npy', data)
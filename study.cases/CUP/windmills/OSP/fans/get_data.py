# r_a : bot / top => [-1, 1]
# r_f : freq2 / freq1 => [0,4]

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

def getRatiosFromName(name):
    # name has format A*.*_f*.*
    cleaned_name = name.replace("A", "").replace("f", "")
    ind_ = cleaned_name.index('_')
    # print(cleaned_name)
    amplitude = float(cleaned_name[:ind_])
    frequency = float(cleaned_name[ind_+1:])
    # return amplitude, frequency
    r_a = amplitude / 2.0
    r_f = frequency / 0.5
    return r_a, r_f


folder = '/scratch/snx3000/anoca/CUP2D/OSP/'
# subfolders_path = [ f.path for f in os.scandir(folder) if f.is_dir() ]
subfolders_name = [os.path.basename(f.path) for f in os.scandir(folder) if f.is_dir()]

num_sims = len(subfolders_name)

sorted_subfolders_name = [[]]
count1 = 0
count2 = 0

def sortFolders(name):
    # given name of folder, returns a number that allows to order the folders
    r_a, r_f = getRatiosFromName(name)
    return r_a

#####
for f in sorted(subfolders_name, key=sortFolders):
    sorted_subfolders_name[count1].append(f)
    count2 += 1

    if count2 >= 21:
        sorted_subfolders_name.append([])
        sorted_subfolders_name[count1] = sorted(sorted_subfolders_name[count1])
        count1 += 1
        count2 = 0
        

sorted_subfolders_name.pop()
print(sorted_subfolders_name[4])

##### save the ratios file

# size is : 41 x 21 x 2
ratios = np.zeros((41, 21, 2))
for index, subf in enumerate(sorted_subfolders_name):
    for index2, subf2 in enumerate(subf):
        r_a, r_f = getRatiosFromName(subf2)
        ratios[index, index2, 0] = r_a
        ratios[index, index2, 1] = r_f

print(ratios.shape)
# save the ratios to a numpy file
np.save('data/ratios.npy', ratios)

##### save the data file

# size is num_sims x num_steps x num_fields x grid_size_x x grid_size_y

data = np.zeros((41, 21, 61, 4, 32, 12)) # we only take the second half of the data in the x direction

for index, list_folder in enumerate(sorted_subfolders_name):
    print(index)
    for index2, folder_name in enumerate(list_folder):
        count_tmp = 0
        count_pres = 0
        count_velx = 0
        count_vely = 0

        
        for file in sorted(os.listdir(folder + folder_name)):
            if file.endswith("-uniform.h5"):
                if file.startswith('tmp_'):
                    fin = h5py.File(folder + folder_name + '/' + file, 'r')
                    data[index, index2, count_tmp, 0, :, :] = fin['data'][0, :, 12:]
                    count_tmp+=1
                elif file.startswith('pres_'):
                    fin = h5py.File(folder + folder_name + '/' + file, 'r')
                    data[index, index2, count_pres, 1, :, :] = fin['data'][0, :, 12:]
                    count_pres+=1
                elif file.startswith('velX_'):
                    fin = h5py.File(folder + folder_name + '/' + file, 'r')
                    data[index, index2, count_velx, 2, :, :] = fin['data'][0, :, 12:]
                    count_velx+=1
                elif file.startswith('velY_'):
                    fin = h5py.File(folder + folder_name + '/' + file, 'r')
                    data[index, index2, count_vely, 3, :, :] = fin['data'][0, :, 12:]
                    count_vely+=1

# # plt.figure()

# # plt.imshow(data[0, 2, 0])

# # plt.savefig('tmp3.png')
np.save('data/ordered_data.npy', data)



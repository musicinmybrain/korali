import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
from numpy.random import normal, multivariate_normal, uniform
import time

from mpi4py import MPI
##################################################################################################################################################################################

# 1) #####----- load the data -----#####

# the dimension of the data vector is only 1

# # size is num_sims_a x numsims_f x num_steps x num_fields x grid_size_x x grid_size_y
# # 41 x 21 x 61 x 4 x 32 x 12
data = np.load('data/ordered_data.npy')
s = data.shape

# 41 x 21 x 2, r_a and r_f
ratios = np.load('data/ratios.npy')

delta_a = ratios[1, 0, 0] - ratios[0, 0, 0]
delta_f = ratios[0, 1, 1] - ratios[0, 0, 1]


# 2) #####----- create the vector of data + noise -----#####

data_shape = data.shape # shape is 41 x 21 x 61 x 4 x 32 x 12
data_shape = data.shape # shape is 41 x 21 x 61 x 4 x 32 x 12
N_a = data.shape[0]
N_b = data.shape[1]
N_step = data.shape[2]
N_fields = data.shape[3]
N_y = data.shape[4]
N_x = data.shape[5]

# compute the standard deviations of the data, one std per field
# std = np.std(data, axis=(0, 1, 2, 4, 5)) # we get a vector with std for each field # size is 4
# # sigma = 1e-2
# sigmas = std.reshape((1, 1, N_fields, 1, 1))
# print(sigmas)
std = 1e-4*np.ones((4))
sigmas = std.reshape((1, 1, N_fields, 1, 1))

# noise = normal(0, scale=sigma, size=data_shape)
cov = np.diag(std)
noise = multivariate_normal(mean=np.zeros(N_fields), cov=cov, size = (N_a, N_b, N_step, N_y, N_x)).reshape(data_shape)

# compute the standard deviations of the data, one std per grid point
# std = np.std(data, axis=(0, 1, 2)) # we get a vector with std for each field, and each grid point # size is 4 x 32 x 12
# # sigma = 1e-2
# sigmas = std.reshape((1, 1, N_fields, N_y, N_x))
# print(sigmas)



y = data + noise

# 3) #####----- compute the utility -----#####

def density(y, F, sigmas): # y and F are both of size : num_sims x num_steps x n
    sqr = 1 / np.sqrt(2* np.pi * sigmas * sigmas)
    temp = y-F
    exp = np.exp(-0.5 * (1/sigmas * sigmas) *temp*temp)
    return sqr * exp

# Ny are the number of points that correspond to one combination of ratios
# => 61 if we take all the simulations and 1 if we do per simulation
factor = delta_a * delta_f / (4 * N_a * N_b * N_step)

sum = np.zeros((N_fields, N_y, N_x))
t1 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 
t2 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 
t3 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 
t4 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if (rank == 0):
    print(size)

num_el = N_step // size     # number of elements per process
start_index = rank * num_el
end_index = (1+rank) * num_el

print(f"Rank {rank} deals with {end_index - start_index} elements.")


t_bef = time.time()
# vectorized method with MPI, but only one type of data, return data with the utility per simulation step, easy to get the utility for all the steps by simply summing
# standard deviation used is different for each data
for m in range(start_index, end_index): # used to be Ny
    # print(m)
    y_1_1 = y[:-1, :-1, m, :, :, :]
    y_0_1 = y[1:, :-1, m, :, :, :]
    y_1_0 = y[:-1, 1:, m, :, :, :]
    y_0_0 = y[1:, 1:, m, :, :, :]

    t1 = density(y_1_1, data[:-1, :-1, m, :, :, :], sigmas)
    t2 = density(y_0_1, data[1:, :-1, m, :, :, :], sigmas)
    t3 = density(y_1_0, data[:-1, 1:, m, :, :, :], sigmas)
    t4 = density(y_0_0, data[1:, 1:, m, :, :, :], sigmas)

    t_1 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 
    t_2 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 
    t_3 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 
    t_4 = np.zeros((N_a-1, N_b-1, N_fields, N_y, N_x)); 

    sum[:, :, :] += factor * np.sum(np.log(t1) + np.log(t2) + np.log(t3) + np.log(t4), axis = (0, 1))
    for i_ in range(1, N_a):
        for j_ in range(1, N_b):
            t_1 += factor * ( density(y_1_1, data[i_-1, j_-1, m, :, :, :], sigmas) + density(y_1_1, data[i_, j_-1, m, :, :, :], sigmas) + \
                                density(y_1_1, data[i_-1, j_, m, :, :, :], sigmas) + density(y_1_1, data[i_, j_, m, :, :, :], sigmas) )

            t_2 += factor * ( density(y_1_0, data[i_-1, j_-1, m, :, :, :], sigmas) + density(y_1_0, data[i_, j_-1, m, :, :, :], sigmas) + \
                                density(y_1_0, data[i_-1, j_, m, :, :, :], sigmas) + density(y_1_0, data[i_, j_, m, :, :, :], sigmas) )

            t_3 += factor * ( density(y_0_1, data[i_-1, j_-1, m, :, :, :], sigmas) + density(y_0_1, data[i_, j_-1, m, :, :, :], sigmas) + \
                                density(y_0_1, data[i_-1, j_, m, :, :, :], sigmas) + density(y_0_1, data[i_, j_, m, :, :, :], sigmas) )

            t_4 += factor * ( density(y_0_0, data[i_-1, j_-1, m, :, :, :], sigmas) + density(y_0_0, data[i_, j_-1, m, :, :, :], sigmas) + \
                                density(y_0_0, data[i_-1, j_, m, :, :, :], sigmas) + density(y_0_0, data[i_, j_, m, :, :, :], sigmas) )

    sum[:, :, :] -= factor * np.sum(np.log(t_1) + np.log(t_2) + np.log(t_3) + np.log(t_4), axis = (0, 1))

# gather all the results from all the processes onto a single process
sum_recv = comm.gather(sum, root = 0)

if rank == 0:
    sum_final = np.zeros((data_shape[2], data_shape[3], data_shape[4], data_shape[5]))

    for i in range(size):
        sum_final[i, :, :, :] = sum_recv[i]

    np.save('results_61.npy', sum_final)

t_aft = time.time()

print(f"simulation took : {t_aft-t_bef} seconds")


##################################################################################################################################################################################
""" 
# 1) #####----- load the data -----#####

# the dimension of the data vector is only 1

# # size is num_sims_a x numsims_f x num_steps x num_fields x grid_size_x x grid_size_y
# # 41 x 21 x 61 x 4 x 32 x 12
data = np.load('ordered_data.npy')
s = data.shape

# 41 x 21 x 2, r_a and r_f
ratios = np.load('ratios.npy')

delta_a = ratios[1, 0, 0] - ratios[0, 0, 0]
delta_f = ratios[0, 1, 1] - ratios[0, 0, 1]


# 2) #####----- create the vector of data + noise -----#####

data_shape = data.shape # shape is 41 x 21 x 61 x 4 x 32 x 12
data_shape = data.shape # shape is 41 x 21 x 61 x 4 x 32 x 12
N_a = data.shape[0]
N_b = data.shape[1]
N_step = data.shape[2]
N_fields = data.shape[3]
N_y = data.shape[4]
N_x = data.shape[5]

# compute the standard deviations of the data
# std = np.std(data, axis=(0, 1, 2, 4, 5)) # we get a vector with std for each field
sigma = 1e-2

noise = normal(0, scale=sigma, size=data.shape)
y = data + noise

# 3) #####----- compute the utility -----#####

def density(y, F, sigma): # y and F are both of size : num_sims x num_steps x n
    sqr = 1 / np.sqrt(2* np.pi * sigma**2)
    temp = y-F
    exp = np.exp(-0.5 * (1/sigma**2) *temp*temp)
    return sqr * exp

# Ny are the number of points that correspond to one combination of ratios
# => 61 if we take all the simulations and 1 if we do per simulation
factor = delta_a * delta_f / (4 * N_a * N_b * N_step)

sum = np.zeros((data_shape[3], data_shape[4], data_shape[5]))
t1 = np.zeros((40, 20, 4, 32, 12)); 
t2 = np.zeros((40, 20, 4, 32, 12)); 
t3 = np.zeros((40, 20, 4, 32, 12)); 
t4 = np.zeros((40, 20, 4, 32, 12)); 


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if (rank == 0):
    print(size)

num_el = N_step // size     # number of elements per process
start_index = rank * num_el
end_index = (1+rank) * num_el

print(f"Rank {rank} deals with {end_index - start_index} elements.")


t_bef = time.time()
# vectorized method with MPI, but only one type of data, return data with the utility per simulation step, easy to get the utility for all the steps by simply summing
# standard deviation used was same for all data : sigma = 1e-2
for m in range(start_index, end_index): # used to be Ny
    # print(m)
    y_1_1 = y[:-1, :-1, m, :, :, :]
    y_0_1 = y[1:, :-1, m, :, :, :]
    y_1_0 = y[:-1, 1:, m, :, :, :]
    y_0_0 = y[1:, 1:, m, :, :, :]

    t1 = density(y_1_1, data[:-1, :-1, m, :, :, :], sigma)
    t2 = density(y_0_1, data[1:, :-1, m, :, :, :], sigma)
    t3 = density(y_1_0, data[:-1, 1:, m, :, :, :], sigma)
    t4 = density(y_0_0, data[1:, 1:, m, :, :, :], sigma)

    t_1 = np.zeros((40, 20, 4, 32, 12)); 
    t_2 = np.zeros((40, 20, 4, 32, 12)); 
    t_3 = np.zeros((40, 20, 4, 32, 12)); 
    t_4 = np.zeros((40, 20, 4, 32, 12)); 

    sum[:, :, :] += factor * np.sum(np.log(t1) + np.log(t2) + np.log(t3) + np.log(t4), axis = (0, 1))
    for i_ in range(1, N_a):
        for j_ in range(1, N_b):
            t_1 += factor * ( density(y_1_1, data[i_-1, j_-1, m, :, :, :], sigma) + density(y_1_1, data[i_, j_-1, m, :, :, :], sigma) + \
                                density(y_1_1, data[i_-1, j_, m, :, :, :], sigma) + density(y_1_1, data[i_, j_, m, :, :, :], sigma) )

            t_2 += factor * ( density(y_1_0, data[i_-1, j_-1, m, :, :, :], sigma) + density(y_1_0, data[i_, j_-1, m, :, :, :], sigma) + \
                                density(y_1_0, data[i_-1, j_, m, :, :, :], sigma) + density(y_1_0, data[i_, j_, m, :, :, :], sigma) )

            t_3 += factor * ( density(y_0_1, data[i_-1, j_-1, m, :, :, :], sigma) + density(y_0_1, data[i_, j_-1, m, :, :, :], sigma) + \
                                density(y_0_1, data[i_-1, j_, m, :, :, :], sigma) + density(y_0_1, data[i_, j_, m, :, :, :], sigma) )

            t_4 += factor * ( density(y_0_0, data[i_-1, j_-1, m, :, :, :], sigma) + density(y_0_0, data[i_, j_-1, m, :, :, :], sigma) + \
                                density(y_0_0, data[i_-1, j_, m, :, :, :], sigma) + density(y_0_0, data[i_, j_, m, :, :, :], sigma) )

    sum[:, :, :] -= factor * np.sum(np.log(t_1) + np.log(t_2) + np.log(t_3) + np.log(t_4), axis = (0, 1))

# gather all the results from all the processes onto a single process
sum_recv = comm.gather(sum, root = 0)

if rank == 0:
    sum_final = np.zeros((data_shape[2], data_shape[3], data_shape[4], data_shape[5]))

    for i in range(size):
        sum_final[i, :, :, :] = sum_recv[i]

    np.save('results_61.npy', sum_final)

t_aft = time.time()

print(f"simulation took : {t_aft-t_bef} seconds")
 """

# t_bef = time.time()
# # vectorized method with MPI, but only one type of data, return data with the utility per simulation step, easy to get the utility for all the steps by simply summing
# for m in range(start_index, end_index): # used to be Ny
#     # print(m)
#     y_1_1 = y[:-1, :-1, m, :, :]
#     y_0_1 = y[1:, :-1, m, :, :]
#     y_1_0 = y[:-1, 1:, m, :, :]
#     y_0_0 = y[1:, 1:, m, :, :]

#     t1 = density(y_1_1, data[:-1, :-1, m, :, :], sigma)
#     t2 = density(y_0_1, data[1:, :-1, m, :, :], sigma)
#     t3 = density(y_1_0, data[:-1, 1:, m, :, :], sigma)
#     t4 = density(y_0_0, data[1:, 1:, m, :, :], sigma)

#     t_1 = np.zeros((40, 20, 32, 12)); 
#     t_2 = np.zeros((40, 20, 32, 12)); 
#     t_3 = np.zeros((40, 20, 32, 12)); 
#     t_4 = np.zeros((40, 20, 32, 12)); 

#     sum[:, :] += factor * np.sum(np.log(t1) + np.log(t2) + np.log(t3) + np.log(t4), axis = (0, 1))
#     for i_ in range(1, N_a):
#         for j_ in range(1, N_b):
#             t_1 += factor * ( density(y_1_1, data[i_-1, j_-1, m, :, :], sigma) + density(y_1_1, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_1_1, data[i_-1, j_, m, :, :], sigma) + density(y_1_1, data[i_, j_, m, :, :], sigma) )

#             t_2 += factor * ( density(y_1_0, data[i_-1, j_-1, m, :, :], sigma) + density(y_1_0, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_1_0, data[i_-1, j_, m, :, :], sigma) + density(y_1_0, data[i_, j_, m, :, :], sigma) )

#             t_3 += factor * ( density(y_0_1, data[i_-1, j_-1, m, :, :], sigma) + density(y_0_1, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_0_1, data[i_-1, j_, m, :, :], sigma) + density(y_0_1, data[i_, j_, m, :, :], sigma) )

#             t_4 += factor * ( density(y_0_0, data[i_-1, j_-1, m, :, :], sigma) + density(y_0_0, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_0_0, data[i_-1, j_, m, :, :], sigma) + density(y_0_0, data[i_, j_, m, :, :], sigma) )

#     sum[:, :] -= factor * np.sum(np.log(t_1) + np.log(t_2) + np.log(t_3) + np.log(t_4), axis = (0, 1))

# print()
# # after all the simulations have run, need to get the data onto a single rank in order to save it
# # sum_recv = [sum]
# sum_recv = comm.gather(sum, root = 0)

# if rank == 0:
#     print(len(sum_recv)) # size 36 # basically 36  61 x 32 x 12 array, but only 1 out of the 61 is non zero, need to concatenate and such
#     print(sum_recv[0].shape) # this is an array
#     print(sum_recv)
#     sum_final = np.zeros((data_shape[2], data_shape[3], data_shape[4]))

#     for i in range(size):
#         sum_final[i, :, :] = sum_recv[i]

#     print(sum_final)
#     np.save('results_36.npy', sum_final)



# # np.save('results.npy', sum)
# t_aft = time.time()

# print(f"simulation took : {t_aft-t_bef} seconds")



# t_bef = time.time()
# # vectorized method, still slow, takes about 1.5 min per iteration, around 60 iterations so 1.5 hours 
# for m in range(1, 2): # used to be Ny
#     print(m)
#     y_1_1 = y[:-1, :-1, m, :, :]
#     y_0_1 = y[1:, :-1, m, :, :]
#     y_1_0 = y[:-1, 1:, m, :, :]
#     y_0_0 = y[1:, 1:, m, :, :]

#     t1 = density(y_1_1, data[:-1, :-1, m, :, :], sigma)
#     t2 = density(y_0_1, data[1:, :-1, m, :, :], sigma)
#     t3 = density(y_1_0, data[:-1, 1:, m, :, :], sigma)
#     t4 = density(y_0_0, data[1:, 1:, m, :, :], sigma)

#     t_1 = np.zeros((40, 20, 32, 12)); 
#     t_2 = np.zeros((40, 20, 32, 12)); 
#     t_3 = np.zeros((40, 20, 32, 12)); 
#     t_4 = np.zeros((40, 20, 32, 12)); 

#     sum += factor * np.sum(np.log(t1) + np.log(t2) + np.log(t3) + np.log(t4), axis = (0, 1))
#     for i_ in range(1, N_a):
#         for j_ in range(1, N_b):
#             t_1 += factor * ( density(y_1_1, data[i_-1, j_-1, m, :, :], sigma) + density(y_1_1, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_1_1, data[i_-1, j_, m, :, :], sigma) + density(y_1_1, data[i_, j_, m, :, :], sigma) )

#             t_2 += factor * ( density(y_1_0, data[i_-1, j_-1, m, :, :], sigma) + density(y_1_0, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_1_0, data[i_-1, j_, m, :, :], sigma) + density(y_1_0, data[i_, j_, m, :, :], sigma) )

#             t_3 += factor * ( density(y_0_1, data[i_-1, j_-1, m, :, :], sigma) + density(y_0_1, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_0_1, data[i_-1, j_, m, :, :], sigma) + density(y_0_1, data[i_, j_, m, :, :], sigma) )

#             t_4 += factor * ( density(y_0_0, data[i_-1, j_-1, m, :, :], sigma) + density(y_0_0, data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y_0_0, data[i_-1, j_, m, :, :], sigma) + density(y_0_0, data[i_, j_, m, :, :], sigma) )

#     sum -= factor * np.sum(np.log(t_1) + np.log(t_2) + np.log(t_3) + np.log(t_4), axis = (0, 1))

# np.save('results.npy', sum)
# t_aft = time.time()

# print(f"simulation took : {t_aft-t_bef} seconds")





# # vectorized method, still slow, takes about 1.5 min per iteration
# for m in range(1, 2): # used to be Ny
#     print(m)

#     t1 = density(y[:-1, :-1, m, :, :], data[:-1, :-1, m, :, :], sigma)
#     t2 = density(y[1:, :-1, m, :, :], data[1:, :-1, m, :, :], sigma)
#     t3 = density(y[:-1, 1:, m, :, :], data[:-1, 1:, m, :, :], sigma)
#     t4 = density(y[1:, 1:, m, :, :], data[1:, 1:, m, :, :], sigma)

#     t_1 = np.zeros((40, 20, 32, 12)); 
#     t_2 = np.zeros((40, 20, 32, 12)); 
#     t_3 = np.zeros((40, 20, 32, 12)); 
#     t_4 = np.zeros((40, 20, 32, 12)); 

#     sum += factor * np.sum(np.log(t1) + np.log(t2) + np.log(t3) + np.log(t4), axis = (0, 1))
#     for i_ in range(1, N_a):
#         for j_ in range(1, N_b):
#             t_1 += factor * ( density(y[:-1, :-1, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[:-1, :-1, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y[:-1, :-1, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[:-1, :-1, m, :, :], data[i_, j_, m, :, :], sigma) )

#             t_2 += factor * ( density(y[:-1, 1:, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[:-1, 1:, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y[:-1, 1:, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[:-1, 1:, m, :, :], data[i_, j_, m, :, :], sigma) )

#             t_3 += factor * ( density(y[1:, :-1, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[1:, :-1, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y[1:, :-1, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[1:, :-1, m, :, :], data[i_, j_, m, :, :], sigma) )

#             t_4 += factor * ( density(y[1:, 1:, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[1:, 1:, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                 density(y[1:, 1:, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[1:, 1:, m, :, :], data[i_, j_, m, :, :], sigma) )

#     sum -= factor * np.sum(np.log(t_1) + np.log(t_2) + np.log(t_3) + np.log(t_4), axis = (0, 1))

# np.save('results.npy', sum)






# # fully un-vectorized method, slow af, takes 5min per iteration of first for loop => 41 * 5 min in total
# for i in range(1, N_a):
#     print(i)
#     for j in range(1, N_b):
#         for m in range(1, N_y):
#             t1 = np.log(density(y[i-1, j-1, m, :, :], data[i-1, j-1, m, :, :], sigma))
#             t2 = np.log(density(y[i, j-1, m, :, :], data[i, j-1, m, :, :], sigma))
#             t3 = np.log(density(y[i-1, j, m, :, :], data[i-1, j, m, :, :], sigma))
#             t4 = np.log(density(y[i, j, m, :, :], data[i, j, m, :, :], sigma))

#             t_1 = 0
#             t_2 = 0
#             t_3 = 0
#             t_4 = 0

#             for i_ in range(1, N_a):
#                 for j_ in range(1, N_b):
#                     t_1 += factor * ( density(y[i-1, j-1, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[i-1, j-1, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                       density(y[i-1, j-1, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[i-1, j-1, m, :, :], data[i_, j_, m, :, :], sigma) )

#                     t_2 += factor * ( density(y[i-1, j, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[i-1, j, m, :, :], data[i, j_-1, m, :, :], sigma) + \
#                                       density(y[i-1, j, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[i-1, j, m, :, :], data[i, j_, m, :, :], sigma) )

#                     t_3 += factor * ( density(y[i, j-1, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[i, j-1, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                       density(y[i, j-1, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[i, j-1, m, :, :], data[i_, j_, m, :, :], sigma) )

#                     t_4 += factor * ( density(y[i, j, m, :, :], data[i_-1, j_-1, m, :, :], sigma) + density(y[i, j, m, :, :], data[i_, j_-1, m, :, :], sigma) + \
#                                       density(y[i, j, m, :, :], data[i_-1, j_, m, :, :], sigma) + density(y[i, j, m, :, :], data[i_, j_, m, :, :], sigma) )

#             sum += factor * ( t1 + t2 + t3 + t4 - ( np.log(t_1) + np.log(t_2) + np.log(t_3) + np.log(t_4) ) )

# np.save('results.npy', sum)









"""
# 1) #####----- load the data -----#####

# # size is num_sims x num_steps x num_fields x grid_size_x x grid_size_y
# # 861 x 61 x 4 x 32 x 12
# data = np.load('data_half.npy')

# the design space are the last two dimensions, hence at the end, they should be the only one remaining

# we play with only the vorticity first
data = np.load('data.npy')[:, :, 0, :, :]
data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))

# 861 x 2, r_a and r_f
ratios = np.load('ratios.npy')

# 2) #####----- create the vector of data + noise -----#####

def covMatrix(n, sigma):
    # can make a fancy covariance matrix
    # rn use the identity
    return sigma * sigma * np.eye(n)

n = data.shape[2] # 384
sigma = 1e-2
E = covMatrix(n, sigma)
detE = lg.det(E)
invE = lg.inv(E)


def noise(n, num_sims, num_steps, cov):
    return multivariate_normal(np.zeros(n), cov, size=(num_sims, num_steps)) # this outputs an array of size .. x n, with n the size of the matrix

y = data + noise(n, data.shape[0], data.shape[1], E)


# 3) #####----- compute the utility -----#####

def density(y, F, sigma, n, det, inv, d): # y and F are both of size : num_sims x num_steps x n
    sqr = 1 / np.sqrt(2* np.pi * det)
    temp = y-F
    exp = np.exp(-0.5 * (1/sigma**2) *temp*temp)
    return sqr * exp

"""

    


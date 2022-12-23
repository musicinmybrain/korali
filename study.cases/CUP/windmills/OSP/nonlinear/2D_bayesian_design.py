import numpy as np
from numpy.random import normal, uniform
import matplotlib.pyplot as plt
import time
from mpi4py import MPI


def sample_theta(size_n, d, which_prior):
    
    e = np.sqrt( (1-np.exp(-0.8))/2.88)
    if which_prior == 1 :
        return uniform(0, 1, size=(size_n, d, d))
    elif which_prior == 2 :
        return uniform(0, e, size=(size_n, d, d))
    elif which_prior == 3 :
        return uniform(e, 1, size=(size_n, d, d))

    # returns n x d x d


def model(thetas, ds):
    # theta : size n x d x d
    # ds : size 2 x d x d
    (n, d, d) = thetas.shape
    res = np.zeros((2, n, d, d))

    res[0, :, :, :] = thetas * thetas * thetas * ds[0] * ds[0] + thetas * np.exp(-np.abs(0.2 - ds[0]))
    res[1, :, :, :] = thetas * thetas * thetas * ds[1] * ds[1] + thetas * np.exp(-np.abs(0.2 - ds[1]))
    # res[0] = thetas * thetas * thetas * d1 * d1 + thetas * np.exp(-np.abs(0.2 - d1))
    # res[1] = thetas * thetas * thetas * d2 * d2 + thetas * np.exp(-np.abs(0.2 - d2))
    return res

    # output of size n x 2 x d x d

def data(thetas, ds, sigma):
    (n, d, d) = thetas.shape
    m = model(thetas, ds) # size n x d
    return m + normal(loc=0, scale=sigma, size=(2, n, d, d)), m

def likelihood(ys, mus, sigma): # for the first term
    # for each y compute likelihood with corresponding theta
    (x, n, d, d) = ys.shape


    # ys is of dimension 2 x n x d x d, so is mus
    # (y - mus)^T (y - mus) should be of size n x d x d 
    temp = ys - mus
    # cste = (1/np.sqrt( (2*np.pi)**2*sigma**4))
    cste = 1 / (2*np.pi * sigma * sigma)

    prod = np.sum(temp * temp, axis = 0)
    res = cste * np.exp(-prod/(2*sigma*sigma))
    return res


    res = cste * np.exp(-np.tensordot(temp, temp, axes=([0], [0]))/ (2*sigma*sigma))
    return res # n x d x d

def likelihood2(ys, mus, sigma): # for the second term
    # we only have one y and compute the likelihood of that y for all theta

    (x, n, d1, d2) = ys.shape

    vec = np.zeros((n, d1, d2))

    for i in range(n):
        temp = (ys-mus[:, i:i+1, :, :])
        cste = 1 / (2*np.pi * sigma * sigma)
        prod = np.sum(temp * temp, axis = 0)
        res = cste * np.exp(-prod/(2*sigma*sigma))
        # print(res.shape)

        # res = (1/np.sqrt((2*np.pi)**2*sigma**4)) * np.exp(-np.tensordot(temp, temp, axes=([0], [0])) / (2*sigma*sigma))
        vec += res
    return vec / n

def utility(ys, mus, sigma):
    # (n, d) = thetas.shape

    first = np.log(likelihood(ys, mus, sigma)) #  n x d x d

    second = np.log(likelihood2(ys, mus, sigma)) # n x d x d

    total = np.mean(first - second, axis=0)
    return total


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if (rank == 0):
    print(size)



### making the whole thing more efficient
n = 5000
sigma = 1e-2
size_d = 101
d1s = np.linspace(0, 1, size_d)
d2s = np.linspace(0, 1, size_d)
ds = np.array(np.meshgrid(d1s, d2s)) # size 2 x d x d

num_el = size_d // (size-1)      # number of elements per process
start_index = rank * num_el
end_index = (1+rank) * num_el

if (rank == size -1):
    end_index = size_d

print(f"Rank {rank} deals with {end_index - start_index} elements.")

time1 = time.time()
########################
# sample theta, d x d times => (n x d x d) matrix
t1 = sample_theta(n, size_d, 1)

# generate data (2 x n x d x d)
ys, mus = data(t1, ds, sigma)

U1 = utility(ys[:, :, :, start_index:end_index], mus[:, :, :, start_index:end_index], sigma)

U1_recv = [U1]
U1_recv = comm.gather(U1_recv, root = 0)


########################
# sample theta, d x d times => (n x d x d) matrix
t2 = sample_theta(n, size_d, 2)

# generate data (2 x n x d x d)
ys, mus = data(t2, ds, sigma)

U2 = utility(ys[:, :, :, start_index:end_index], mus[:, :, :, start_index:end_index], sigma)

U2_recv = [U2]
U2_recv = comm.gather(U2_recv, root = 0)


########################
# sample theta, d x d times => (n x d x d) matrix
t3 = sample_theta(n, size_d, 3)

# generate data (2 x n x d x d)
ys, mus = data(t3, ds, sigma)

U3 = utility(ys[:, :, :, start_index:end_index], mus[:, :, :, start_index:end_index], sigma)

U3_recv = [U3]
U3_recv = comm.gather(U3_recv, root = 0)

time2 = time.time()
print(time2-time1)
if rank == 0:
    plt.figure(1)
    U1_final = U1_recv[:][0][0]

    for i in range(1, size):
        U1_final = np.concatenate([U1_final, U1_recv[:][i][0]], axis=1)
    
    plt.contour(ds[0], ds[1], U1_final)
    plt.colorbar()

    plt.savefig('2D_utility_1.png')


    plt.figure(2)
    U2_final = U2_recv[:][0][0]

    for i in range(1, size):
        U2_final = np.concatenate([U2_final, U2_recv[:][i][0]], axis=1)
    
    plt.contour(ds[0], ds[1], U2_final)
    plt.colorbar()

    plt.savefig('2D_utility_2.png')


    plt.figure(3)
    U3_final = U3_recv[:][0][0]

    for i in range(1, size):
        U3_final = np.concatenate([U3_final, U3_recv[:][i][0]], axis=1)
    
    plt.contour(ds[0], ds[1], U3_final)
    plt.colorbar()

    plt.savefig('2D_utility_3.png')




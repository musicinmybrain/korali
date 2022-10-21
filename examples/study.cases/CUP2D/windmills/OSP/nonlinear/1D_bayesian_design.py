import numpy as np
from numpy.random import normal, uniform
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

def sample_theta(size_n, d, which_prior):
    e = np.sqrt( (1-np.exp(-0.8))/2.88)
    if which_prior == 1 :
        return uniform(0, 1, size=(size_n, d))
    elif which_prior == 2 :
        return uniform(0, e, size=(size_n, d))
    elif which_prior == 3 :
        return uniform(e, 1, size=(size_n, d))


def model(thetas, d): # G
    res = thetas * thetas * thetas * d * d + thetas * np.exp(-np.abs(0.2 - d))
    return res

def data(thetas, ds, sigma):
    (n, d) = thetas.shape
    m = model(thetas, ds) # size n x d
    return m + normal(loc=0, scale=sigma, size=(n, d)), m

def likelihood(ys, mus, sigma): # for the first term
    # for each y compute likelihood with corresponding theta
    res = (1/np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-(ys-mus)*(ys-mus) / (2*sigma*sigma))
    return res # n x d

def likelihood2(ys, mus, sigma): # for the second term
    # we only have one y and compute the likelihood of that y for all theta

    (n, d) = ys.shape

    # #tmpy = np.ones((n, n, d)) * ys[:, np.newaxis, :]
    # tmpy = np.tile(ys[:, np.newaxis, :], (1, n, 1))
    # diff = tmpy - mus[np.newaxis, :, :]

    # res = (1/np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-np.multiply(diff, diff) / (2*sigma*sigma))
    # return np.mean(res, axis=0)

    vec = np.zeros((n, d))

    for i in range(n):
        temp = ys - mus[i]
        res = (1/np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-np.multiply(temp, temp) / (2*sigma*sigma))
        vec += res
    return vec / n


    # log likelihood trick
    cste = np.log(1/np.sqrt(2*np.pi*sigma*sigma))

    # compute the value of the term
    for i in range(n):
        temp = ys - mus[i]
        term = -(temp * temp)/ (2*sigma*sigma)




def utility(thetas, ys, ds, mus, sigma):
    (n, d) = thetas.shape

    
    first = np.log(likelihood(ys, mus, sigma))
    

    second = np.log(likelihood2(ys, mus, sigma))
    


    total = np.mean(first - second, axis=0)
    return total


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if (rank == 0):
    print(size)


### making the whole thing more efficient
n = 400
sigma = 1e-2
size_d = 101
ds = np.linspace(0, 1, size_d).reshape((1, size_d)) # size 1 x d

# depending on the size, we split the problem into a number of subproblems equal to the size 
# each computation is independent of the design space, so parallelize in that dimension

num_el = size_d // (size-1)      # number of elements per process
start_index = rank * num_el
end_index = (1+rank) * num_el

if (rank == size -1):
    end_index = size_d

print(f"Rank {rank} deals with {end_index - start_index} elements.")

time1 = time.time()

# sample theta, d times => (n x d) matrix
t1 = sample_theta(n, size_d, 1)

# generate data (n x d)
ys, mus = data(t1, ds, sigma)

# compute the utility
U1 = utility(t1[:, start_index:end_index], ys[:, start_index:end_index], ds[:, start_index:end_index], mus[:, start_index:end_index], sigma) #.reshape((1, 101)) # vector of size 1 x d

U1 = comm.gather(U1, root = 0)

# sample theta, d times => (n x d) matrix
t2 = sample_theta(n, size_d, 2)

# generate data (n x d)
ys, mus = data(t2, ds, sigma)

# compute the utility
U2 = utility(t2[:, start_index:end_index], ys[:, start_index:end_index], ds[:, start_index:end_index], mus[:, start_index:end_index], sigma) #.reshape((1, 101)) # vector of size 1 x d

U2 = comm.gather(U2, root = 0)

# sample theta, d times => (n x d) matrix
t3 = sample_theta(n, size_d, 3)

# generate data (n x d)
ys, mus = data(t3, ds, sigma)

# compute the utility
U3 = utility(t3[:, start_index:end_index], ys[:, start_index:end_index], ds[:, start_index:end_index], mus[:, start_index:end_index], sigma) #.reshape((1, 101)) # vector of size 1 x d

U3 = comm.gather(U3, root = 0)

time2=time.time()
print(f"Rank {rank} took time {time2-time1} to simulate.")


if (rank == 0):
    plt.figure(1)
    U1 = np.concatenate(U1)
    U2 = np.concatenate(U2)
    U3 = np.concatenate(U3)

    plt.plot(ds.reshape((101)), U1)
    plt.plot(ds.reshape((101)), U2)
    plt.plot(ds.reshape((101)), U3)

    plt.xlabel('d')
    plt.ylabel('Utility')

    plt.savefig('1D_utility_test.png')






"""
# compute the utility
#U1 = utility(t1, ys, ds, mus, sigma) #.reshape((1, 101)) # vector of size 1 x d

#print(U1)

# sample theta, d times => (n x d) matrix
t2 = sample_theta(n, size_d, 2)

# generate data (n x d)
ys, mus = data(t2, ds, sigma)

# compute the utility
U2 = utility(t2, ys, ds, mus, sigma) #.reshape((1, 101)) # vector of size 1 x d
#print(U1)

# sample theta, d times => (n x d) matrix
t3 = sample_theta(n, size_d, 3)

# generate data (n x d)
ys, mus = data(t3, ds, sigma)

# compute the utility
U3 = utility(t3, ys, ds, mus, sigma) #.reshape((1, 101)) # vector of size 1 x d
#print(U1)


plt.figure(1)

plt.plot(ds.reshape((101)), U1.reshape((101)))
plt.plot(ds.reshape((101)), U2.reshape((101)))
plt.plot(ds.reshape((101)), U3.reshape((101)))
# plt.plot(ds, U2)
# plt.plot(ds, U3)

plt.savefig('1D_utility.png')

"""



# design space d = [0, 1]

# prior \theta ~ U (0, 1) = p(\theta)

# y(\theta, d) = G(\theta, d) + \epsilon = \theta^3 * d^2 + \theta * exp(-|0.2 - d|) + \epsilon
# => y ~ N (G(\theta, d), sigma^2), where epsilon ~ N (0, sigma^2)


# n_out = number of samples of the data, 
# y^(i) drawn from p(y| \theta = \theta^(i), d) => sample different y for different theta, given d
# the theta are drawn from the prior p(\theta)

# to compute the value of p(y^(i)| \theta^(i), d)


# p(y^(i)|d) is usually not analytic, so we approximate using important sampling
# p(y^(i)|d) = (1/n_in) * \sum_{j=1}^n_in p(y^(i)| \theta^(i, j), d), 
# \theta^(., j) are drawn from the prior p(\theta)


# U = (1/n_out) * \sum_{i=1}^n_out [ \ln(p(y^(i)| \theta = \theta^(i), d)) - \ln((1/n_in) * \sum_{j=1}^n_in p(y^(i)| \theta^(i, j), d))]

# to mitigate cost of nested MC estimator, draw a fresh batch of prior samples {\theta^(k)}_{k=1}^n_out for every d
# use this set for both outer and inner MC sums, i.e. \theta^(i) = \theta^(k) and \theta^(., j) = \theta^(k) 
# n_out = n_in

# we can now compute the utility



# # step 1 : try out the model, it works

# m = model(thetas, 0.5)
# print(thetas.shape)
# print(m.shape)

# plt.figure(1)

# plt.plot(thetas, m, '.') # the points are not sorted, so plotting with lines between data points is wrong
# plt.plot(ds, model(ds, 0.5))

# plt.savefig('model.png')

# # step 2 : generate data with noise, it works

# ys = data(n, thetas, 0.5, sigma)

# plt.figure(2)

# plt.plot(thetas, ys, 'k.')
# plt.plot(ds, model(ds, 0.5))

# plt.savefig('noisy_data.png')


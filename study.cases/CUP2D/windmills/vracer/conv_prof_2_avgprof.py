import numpy as np
import matplotlib.pyplot as plt

folder = "profiles/"
file_ = ["profile_" + str(i) + ".dat" for i in range(11)]

mean = 0

# for the mean
for ind, file in enumerate(file_):
    data = np.genfromtxt(folder + file, delimiter=' ')

    # divide the cumulative sum of the values by the time => time average essentially
    t = data[:4000, 0:1]
    mean = np.cumsum(data[:4000, 1:], axis = 0) * 0.05 / t
    # np.savetxt("avgprofiles/avg" + file, mean)

    # if(ind == 0):
    #     plt.figure()
    #     plt.plot(t, cumulative_summed[:, 12])
    #     plt.savefig("test.png")

# for the standard deviation, cumulative std

# for ind, file in enumerate(file_):
#     data = np.genfromtxt(folder + file, delimiter=' ')

#     # divide the cumulative sum of the values by the time => time average essentially
#     t = data[:4000, 0:1]
#     variance = np.zeros((4000, 32))

#     for i in range(4000):
#         variance[i] = np.sum((data[:i+1, 1:] - mean[i])**2, axis = 0) * 0.05 / t[i]

#     np.savetxt("avgprofiles/std" + file, variance)

    # if(ind == 6):
    #     plt.figure()
    #     plt.plot(t, mean[:, 12])
    #     plt.savefig("2fans/conv_2fans_02.png")

    #     plt.figure()
    #     plt.plot(np.linspace(1, 32, 32), mean[3999, :])
    #     plt.fill_between(np.linspace(1, 32, 32),  mean[3999, :] - np.sqrt( variance[3999, :]),  mean[3999, :] + np.sqrt(variance[3999, :]), alpha=0.2)
    #     plt.savefig("2fans/prof_2fans_02.png")

# for the standard deviation, final std
all_variances = np.zeros((11, 32))
for ind, file in enumerate(file_):
    data = np.genfromtxt(folder + file, delimiter=' ')

    # divide the cumulative sum of the values by the time => time average essentially
    t = data[:4000, 0:1]

    all_variances[ind] = np.sum((data[:, 1:] - mean[-1])**2, axis = 0) * 0.05 / t[-1]


np.savetxt("avgprofiles/stdprofiles.dat", all_variances)






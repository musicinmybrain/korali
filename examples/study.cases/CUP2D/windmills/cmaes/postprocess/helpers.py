import numpy as np
from scipy.integrate import simps, trapz

def average_profile(data, start, end):

    frequency = 0.5
    period = 1/frequency
    steps_per_period = int(period/0.05)
    num_periods = int(np.floor((end - start + 1) / steps_per_period))

    t = data[:, 0]

    avg_profs = np.zeros((num_periods, 16))

    for i in range(1, num_periods+1):
        index = start + i * steps_per_period
        T = t[start] + i * period
        avg_profs[i-1, :] = (1/(T - t[start])) * trapz(data[start:index + 1, 1:], t[start:index + 1], axis = 0)

    return avg_profs


def mse(data, start, end):
    frequency = 0.5
    period = 1/frequency
    steps_per_period = 40
    num_periods = int(np.floor((end - start + 1) / steps_per_period))

    t = data[:, 0]

    avg_profs = np.zeros((num_periods, 32))

    for i in range(1, num_periods+1):
        index = start + i * steps_per_period
        T = t[start] + i * period
        avg_profs[i-1, :] = (1/(T - t[start])) * trapz(data[start:index + 1, 1:], t[start:index + 1], axis = 0)


    diff = avg_profs[-1, :] - avg_profs[:, :]

    mse = np.sqrt(np.sum(diff * diff, axis = 1))

    return mse
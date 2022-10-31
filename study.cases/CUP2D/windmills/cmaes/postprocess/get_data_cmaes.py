import matplotlib.pyplot as plt
import numpy as np
import json
import os


# data for plotting the landscape of the objective function
# get the target profile:
fol = '../data/'
target_x = np.genfromtxt(fol + 'x_profile.dat', delimiter=' ')
target_y = np.genfromtxt(fol + 'y_profile.dat', delimiter=' ')
target_norm = np.sqrt(np.sum(target_x*target_x + target_y*target_y))

# get the data from the different cmaes simulations
num_sims = 10
names = [f"{i}" for i in range(1, 1 + num_sims)]

# correct=[3, -3, 0.25, 0.5]

means = np.zeros((num_sims, 150, 4))
parameters = np.zeros((num_sims, 150, 8, 4))
values = np.zeros((num_sims, 150, 8, 1))
best_values = np.zeros(num_sims)
optimum_parameters = np.zeros((num_sims, 4))

for index, name in enumerate(names):
    print(index)
    path = '/scratch/snx3000/anoca/korali/cmaes/' + name + '/_results/'

    resultFiles = [
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.startswith('gen')
    ]

    resultFiles = sorted(resultFiles)

    genlist = [None for i in range(len(resultFiles))]

    for file in resultFiles:
        with open(path + file) as f:
            genJs = json.load(f)
            curGen = genJs['Current Generation']
            genlist[curGen] = genJs

    del genlist[0] # remove first useless element

    for ind, gen in enumerate(genlist):
        means[index, ind, :] = gen['Solver']['Current Mean']
        parameters[index, ind, :, :] = gen['Solver']['Sample Population']
        values[index, ind, :, :] = np.array(gen['Solver']['Value Vector']).reshape((8, 1))

    best_values[index] = genlist[-1]['Results']['Best Sample']['F(x)']
    optimum_parameters[index] = genlist[-1]['Results']['Best Sample']['Parameters']

# convert values to log scale, normalized by target profile
logf = lambda x : np.log10(np.sqrt(np.abs(x))/target_norm)
values = logf(values)

# save the cmaes data in a file
np.savez(fol + 'cmaes_data.npz', param=parameters, values=values, means=means, best=best_values, optimum=optimum_parameters)
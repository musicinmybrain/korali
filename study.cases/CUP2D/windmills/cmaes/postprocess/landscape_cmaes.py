import matplotlib.pyplot as plt
import numpy as np
import json
import os
from landscape_gridsearch import plotLandscapeGrid

save_folder = '../plots/'

# load data from cmaes_data.npz
data = np.load('../data/cmaes_data.npz')
parameters = data['param'] # 10 x 150 x 8 x 4
values = data['values'] # 10 x 150 x 8 x 1
means = data['means'] # 10 x 150 x 4
best_values = data['best'] # 10
optimum_parameters = data['optimum'] # 10 x 4




################################### all the points

fig=plt.figure(figsize=(16, 10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(parameters[:, :, :, 0].flatten(), parameters[:, :, :, 1].flatten(), values[:, :, :, 0].flatten(),  marker='.', c=values[:, :, :, 0].flatten(), cmap='plasma')
ax.set_xlabel('A1')
ax.set_ylabel('A2')
ax.set_zlabel('log |F|')
plt.title("A2 vs A1")

#plt.savefig('all.png')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(parameters[:, :, :, 2].flatten(), parameters[:, :, :, 3].flatten(), values[:, :, :, 0].flatten(),  marker='.', c=values[:, :, :, 0].flatten(), cmap='plasma')
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('log |F|')
plt.title("f2 vs f1")

plt.savefig(save_folder + 'all.png')






""" 

# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# top = cm.get_cmap('Blues', 128)
# bottom = cm.get_cmap('YlOrRd', 128)
# newcolors = np.vstack((top(np.linspace(0.75, 1, 128)),
#                        bottom(np.linspace(0.5, 1, 128)[-1::-1])))
# newcmp = ListedColormap(cm.get_cmap('Blues', 128)(np.linspace(0.75, 1, 128)))

# newcmp = ListedColormap(cm.get_cmap('Greens', 128)(np.linspace(0.5, 1, 128)))
# colourMap2 = cm.ScalarMappable(cmap=newcmp)
# colourMap2.set_array(np.linspace(-0.3,0.5,10))
# colBar2 = plt.colorbar(colourMap2, orientation='horizontal',pad = 0).set_label('Veloctiy Vx')

# get the norm of the target profile:
fol = 'results/slowdiff/'
target_x = np.genfromtxt(fol + 'x_profile.dat', delimiter=' ')
target_y = np.genfromtxt(fol + 'y_profile.dat', delimiter=' ')
target_norm = np.sqrt(np.sum(target_x*target_x + target_y*target_y))


# num_sims = 20
# names = [f"{i}" for i in range(1, num_sims + 1)]

# we don't take into account first 11 simulations, wrong run
num_sims = 10
names = [f"{i}" for i in range(1, 1 + num_sims)]

correct=[3, -3, 0.25, 0.5]

means = np.zeros((num_sims, 150, 4))
parameters = np.zeros((num_sims, 150, 8, 4))
values = np.zeros((num_sims, 150, 8, 1))
best_values = np.zeros(num_sims)
optimum_parameters = np.zeros((num_sims, 4))

for index, name in enumerate(names):
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
# loop over all the generations and all the population and obtain the 
# print(genlist[1]['Solver'].keys())

# print(np.array(genlist[-1]['Solver']['Current Mean']))
# print(np.array(genlist[1]['Solver']['Value Vector']))
    # print(genlist[1]['Solver'].keys())

    for ind, gen in enumerate(genlist):
        means[index, ind, :] = gen['Solver']['Current Mean']
        parameters[index, ind, :, :] = gen['Solver']['Sample Population']
        values[index, ind, :, :] = np.array(gen['Solver']['Value Vector']).reshape((8, 1))

    best_values[index] = gen['Results']['Best Sample']['F(x)']
    optimum_parameters[index] = gen['Results']['Best Sample']['Parameters']


# get the best fct value and the corresponding sims
sorted_indices = np.argsort(best_values)
print(sorted_indices)
sorted_results = [(names[int(i)], np.sqrt(np.abs(best_values[int(i)]))/ target_norm) for i in sorted_indices] # name of sim and resulting profile
print(sorted_results)
print



best_sim_indices = sorted_indices[-3:]
best_values_sim = values[best_sim_indices]
best_parameters = parameters[best_sim_indices]

best_means = means[best_sim_indices]
# best_means_f = best_values_sim[]

sec_best_sim_indices = sorted_indices[-6:-3]

sec_best_values_sim = values[sec_best_sim_indices]
sec_best_parameters = parameters[sec_best_sim_indices]

logf = lambda x : np.log10(np.sqrt(np.abs(x))/target_norm)
# logf = lambda x : np.abs(x)

################################### all the points

# create the ratios between the parameters # second fan divided by first fan

r_a = best_parameters[:, :, :, 1] / best_parameters[:, :, :, 0]
r_f = best_parameters[:, :, :, 3] / best_parameters[:, :, :, 2]

fig=plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(r_a.flatten(), r_f.flatten(), logf(best_values_sim[:, :, :, 0].flatten()),  marker='.', c=logf(best_values_sim[:, :, :, 0].flatten()), cmap='plasma')
ax.set_xlabel('ratio amplitude')
ax.set_ylabel('ratio frequency')
ax.set_zlabel('log |F|')
ax.set_xlim([-3, 6])
ax.set_ylim([-10, 10])

plt.savefig('cmaes_best.png')

r_a = parameters[:, :, :, 1] / parameters[:, :, :, 0]
r_f = parameters[:, :, :, 3] / parameters[:, :, :, 2]

r_a_opt = optimum_parameters[:, 1] / optimum_parameters[:, 0]
r_f_opt = optimum_parameters[:, 3] / optimum_parameters[:, 2]

fig=plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(r_a.flatten(), r_f.flatten(), logf(values[:, :, :, 0].flatten()),  marker='.', c=logf(values[:, :, :, 0].flatten()), cmap='plasma')
# ax.set_xlabel('ratio amplitude')
# ax.set_ylabel('ratio frequency')
# ax.set_zlabel('log |F|')

plt.plot(r_a_opt, r_f_opt, 'rx')
plt.xlabel('ratio amplitude')
plt.ylabel('ratio frequency')

plt.xlim([-2., 2.])
plt.ylim([-5, 20])

# ax.view_init(elev=-90)

plt.savefig('cmaes_all.png')
 """
################################### all the points

# fig=plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(parameters[:, :, :, 0].flatten(), parameters[:, :, :, 1].flatten(), logf(values[:, :, :, 0].flatten()),  marker='.', c=logf(values[:, :, :, 0].flatten()))
# ax.set_xlabel('A1')
# ax.set_ylabel('A2')
# ax.set_zlabel('log |F|')
# plt.title("A2 vs A1")

# #plt.savefig('all.png')

# ax = fig.add_subplot(122, projection='3d')
# ax.scatter(parameters[:, :, :, 2].flatten(), parameters[:, :, :, 3].flatten(), logf(values[:, :, :, 0].flatten()),  marker='.', c=logf(values[:, :, :, 0].flatten()))
# ax.set_xlabel('f1')
# ax.set_ylabel('f2')
# ax.set_zlabel('log |F|')
# plt.title("f2 vs f1")

# plt.savefig('all.png')


""" 
####################################
fig2=plt.figure(figsize=(16, 10))
ax = fig2.add_subplot(121, projection='3d')
# , c=logf(best_values_sim[:, :, :, 0].flatten())
ax.scatter(best_parameters[:, :, :, 0].flatten(), best_parameters[:, :, :, 1].flatten(), logf(best_values_sim[:, :, :, 0].flatten()), c = np.arange(1, 1201),  marker='.', cmap=newcmp)
ax.set_xlabel('A1')
ax.set_ylabel('A2')
ax.set_zlabel('log |F|')
plt.title("A2 vs A1")
ax.view_init(elev=10,azim=30)

#plt.savefig('all.png')

ax = fig2.add_subplot(122, projection='3d')
ax.scatter(best_parameters[:, :, :, 2].flatten(), best_parameters[:, :, :, 3].flatten(), logf(best_values_sim[:, :, :, 0].flatten()),  marker='.', cmap='Greens')
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('log |F|')
plt.title("f2 vs f1")

plt.savefig('best.png')


####################################
fig3=plt.figure(figsize=(16, 10))
ax = fig3.add_subplot(121, projection='3d')
ax.scatter(sec_best_parameters[:, :, :, 0].flatten(), sec_best_parameters[:, :, :, 1].flatten(), logf(sec_best_values_sim[:, :, :, 0].flatten()),  marker='.', c=logf(sec_best_values_sim[:, :, :, 0].flatten()))
ax.set_xlabel('A1')
ax.set_ylabel('A2')
ax.set_zlabel('log |F|')
plt.title("A2 vs A1")

#plt.savefig('all.png')

ax = fig3.add_subplot(122, projection='3d')
ax.scatter(sec_best_parameters[:, :, :, 2].flatten(), sec_best_parameters[:, :, :, 3].flatten(), logf(sec_best_values_sim[:, :, :, 0].flatten()),  marker='.', c=logf(sec_best_values_sim[:, :, :, 0].flatten()))
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('log |F|')
plt.title("f2 vs f1")

plt.savefig('2ndbest.png')


####################################
fig2=plt.figure(figsize=(16, 10))
ax = fig2.add_subplot(121)
# , c=logf(best_values_sim[:, :, :, 0].flatten())
ax.scatter(best_means[:, :, 0].flatten(), best_means[:, :, 1].flatten(), c = np.arange(1, 151),  marker='x', cmap=newcmp)
ax.set_xlabel('A1')
ax.set_ylabel('A2')
# ax.set_zlabel('log |F|')
plt.title("A2 vs A1")
# ax.view_init(elev=10,azim=30)

#plt.savefig('all.png')

ax = fig2.add_subplot(122)
ax.scatter(best_means[:, :, 2].flatten(), best_means[:, :, 3].flatten(), c = np.arange(1, 151), marker='x', cmap='plasma')
ax.set_xlabel('f1')
ax.set_ylabel('f2')
# ax.set_zlabel('log |F|')
plt.title("f2 vs f1")

plt.savefig('bestmeans.png') """
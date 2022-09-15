import matplotlib.pyplot as plt
import numpy as np

save_folder = '../plots/'

def plotLandscapeGrid():

    # 1) #####----- load the data -----#####

    data = np.load('../data/final_profiles.npy')# size 71 x 66 x 2 x 16
    ratios = np.load('../data/ratios.npy') # size 71 x 66 x 2

    # the solution data[5][20] for A=-3 f=0.5
    a_index = 5
    f_index = 20
    solution = data[a_index, f_index, :, :]
    norm = np.sqrt(np.sum(solution * solution, axis=(0,1)))

    # 2) #####----- compute the objective function -----#####

    diff = data - solution
    objective_fct = np.sqrt(np.sum(diff * diff, axis=(2, 3))) / norm # 71 x 66 array

    # set the solution point to have a value of 1e-2 so we can see on the plot
    objective_fct[a_index, f_index] = 10**(-2)

    # 3) #####----- compute the objective function -----#####

    X = ratios[:, 15:, 0]
    Y = ratios[:, 15:, 1]
    Z = np.log10(objective_fct[:, 15:])

    fig_contourf, ax_contourf = plt.subplots()

    plt.contourf(X, Y, Z, levels=12, cmap='plasma')
    dot, = ax_contourf.plot(ratios[a_index, f_index, 0], ratios[a_index, f_index, 1], 'rx', label='target')
    ax_contourf.set_aspect('auto')
    dot.set_label('target')
    plt.colorbar()
    plt.xlabel('amplitude ratio')
    plt.ylabel('frequency ratio')
    plt.title('objective function landscape (logscale base 10)')

    fig_surf, ax_surf = plt.subplots(subplot_kw={"projection": "3d"})
    fig_surf.set_size_inches(15, 10)

    surf = ax_surf.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', linewidth=0.3, ccount=66, rcount=71)
    fig_surf.colorbar(surf, ax=ax_surf)
    plt.xlabel('amplitude ratio')
    plt.ylabel('frequency ratio')
    plt.title('objective function landscape (logscale base 10)')

    # ax.view_init(elev=90, azim=0)


    return fig_contourf, ax_contourf, fig_surf, ax_surf


fig1, ax1, fig2, ax2 = plotLandscapeGrid()

plt.figure(fig1.get_label())

plt.savefig(save_folder + 'landscape_contourf.png')
plt.close()

plt.figure(fig2.get_label())

plt.savefig(save_folder + 'landscape_surface.png')
plt.close()
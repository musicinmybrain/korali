import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import numpy as np

def createDomain(ax):
    ax.set_xlim([0, 0.525])
    ax.set_ylim([0, 0.7])
    smajax = 0.0405
    sminax = 0.0135
    frac = 0.55

    d = smajax * (1. -(2./3.) * frac)
    num_fans = 2
    centers = [[0.1, 0.25], [0.1, 0.45]]

    # profile = Rectangle((0.35, 0.175), 0.0875, 0.35)
    # profile.set_facecolor('white')
    # profile.set_edgecolor('black')

    # ax.add_patch(profile)

    for i in range(num_fans):
        # bottom ellipse
        center1 = [d * np.sin(np.pi/6), -d * np.cos(np.pi/6)]
        bot_ellipse = Ellipse((centers[i][0] + center1[0], centers[i][1] + center1[1]), 2*smajax, 2*sminax, angle=-60)
        bot_ellipse.set_color('gray')
        ax.add_patch(bot_ellipse)

        # top ellipse
        center2 = [d * np.sin(np.pi/6), d * np.cos(np.pi/6)]
        top_ellipse = Ellipse((centers[i][0] + center2[0], centers[i][1] + center2[1]), 2*smajax, 2*sminax, angle=60)
        top_ellipse.set_color('gray')
        ax.add_patch(top_ellipse)

        # top ellipse
        center3 = [-d, 0]
        middle_ellipse = Ellipse((centers[i][0] + center3[0], centers[i][1] + center3[1]), 2*smajax, 2*sminax, angle=0)
        middle_ellipse.set_color('gray')
        ax.add_patch(middle_ellipse)

    ax.set_aspect('equal')

utility = np.load('results_61.npy') # size 61 x 4 x 32 x 12
# utility = np.load('results_61_separate.npy') # size 61 x 4 x 32 x 12
# utility = np.load('results_61_separate_errorless.npy') # size 61 x 4 x 32 x 12

name = 'sep_err'

vorticity = np.sum(utility[:, 0, :, :], axis = 0)
pressure = np.sum(utility[:, 1, :, :], axis = 0)
velx = np.sum(utility[:, 2, :, :], axis = 0)
vely = np.sum(utility[:, 3, :, :], axis = 0)

output_folder = 'plots/'
# vorticities
fig, ax = plt.subplots()

createDomain(ax)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), vorticity)
plt.colorbar()
plt.title('Utility : vorticity')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'vorticity' + '_' + name + '.png')

# pressure
fig2, ax2 = plt.subplots()

createDomain(ax2)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), pressure)
plt.colorbar()
plt.title('Utility : pressure')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'pressure' + '_' + name + '.png')

# pressure
fig3, ax3 = plt.subplots()

createDomain(ax3)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), velx)
plt.colorbar()
plt.title('Utility : velx')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'velx' + '_' + name + '.png')

# pressure
fig4, ax4 = plt.subplots()

createDomain(ax4)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), vely)
plt.colorbar()
plt.title('Utility : vely')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'vely' + '_' + name + '.png')


# # doing the computation of the utility using the combined data
# utility_combined = np.load('results_61_combined.npy') # size 61 x 32 x 12

# all = np.sum(utility_combined, axis=0)

# output_folder = 'plots/'
# # vorticities
# fig, ax = plt.subplots()

# createDomain(ax)

# plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), all)
# plt.colorbar()
# plt.title('Utility : vorticity, pressure, velx, vely')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.savefig(output_folder + 'all.png')
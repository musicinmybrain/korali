import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Ellipse
import numpy as np

fig, ax = plt.subplots()

ax.set_xlim([0, 0.525])
ax.set_ylim([0, 0.7])

profile = Rectangle((0.35, 0.175), 0.0875, 0.35)
profile.set_facecolor('white')
profile.set_edgecolor('black')

ax.add_patch(profile)

smajax = 0.0405
sminax = 0.0135
frac = 0.55

d = smajax * (1. -(2./3.) * frac)
num_fans = 2
centers = [[0.1, 0.25], [0.1, 0.45]]

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


plt.savefig('domain.png')




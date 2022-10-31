#! /usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_file: str,
         png_file: str,
         target_radius: float,
         start_radius: float=1,
         minspace: float=3):
    df = pd.read_csv(csv_file)

    nabfs = len(df.columns) // 2

    if (not png_file is None):
     plt.ioff()
     matplotlib.use('Agg')

    fig, ax = plt.subplots()

    d = minspace
    xmin = -d
    xmax = d
    ymin = -d
    ymax = d

    for i in range(nabfs):
        x = df[f"x{i}"].to_numpy()
        y = df[f"y{i}"].to_numpy()
        ax.plot(x, y)

        start = plt.Circle((x[0], y[0]), start_radius, color='g', alpha = 0.3)
        ax.add_artist(start)

        #xmin = min([xmin, min(x)-d])
        #xmax = max([xmax, max(x)+d])
        #ymin = min([ymin, min(y)-d])
        #ymax = max([ymax, max(y)+d])

        xmin = -20
        xmax = +40
        ymin = -20
        ymax = +40

    target = plt.Circle((0, 0), target_radius, color='r', alpha = 0.3)
    ax.add_artist(target)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.gca().set_aspect('equal', adjustable='box')
    if (png_file is None):
     plt.show() 
    else: 
     plt.savefig(png_file, bbox_inches='tight')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Indicates the input file path.', default = '_result/best.csv', required = False)
    parser.add_argument('--output', help='Indicates the output file path. If not specified, it prints to screen.', required = False)
    parser.add_argument('--target-radius', type=float, default=1)
    args = parser.parse_args()

    plot(args.input, args.output, args.target_radius)

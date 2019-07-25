#! /usr/bin/env python3

import os
import sys
import glob
import time
import json
import colorsys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from korali.plotter.helpers import readFiles, hlsColors, pauseLight, drawMulticoloredLine


# Plot DEA results (read from .json files)
def plot_dea(src, live=False, test=False):

    init     = False # init flag
    gen      = 0  # generation
    numdim   = 0  # problem dimension
    names    = [] # description params
    colors   = [] # rgb colors
    numeval  = [] # number obj function evaluations
    dfval    = [] # abs diff currentBest - bestEver
    fval     = [] # best fval current generation
    fvalXvec = [] # location fval
    meanXvec = [] # location mean population
    width    = [] # spread population

    plt.style.use('seaborn-dark')

    fig = None
    ax  = None
    
    resultfiles = readFiles(src)
   
    for filename in resultfiles:
        path   = '{0}/{1}'.format(src, filename)
 
        with open(path) as f:
            
            data  = json.load(f)
            state = data['Solver']['Internal']
            gen   = data['General']['Current Generation']

            if (init == False):
                
                init = True
                fig, ax = plt.subplots(2,2,num='DEA live diagnostics: {0}'.format(src),figsize=(8,8))
                fig.show()
                
                numdim = len(data['Variables'])
                names  = [ data['Variables'][i]['Name'] for i in range(numdim) ]
                colors = hlsColors(numdim)
                for i in range(numdim):
                    fvalXvec.append([])
                    meanXvec.append([])
                    width.append([])
                continue
                
            numeval.append(data['General']['Function Evaluation Count'])
            dfval.append(abs(state['Current Best Value'] - state['Best Ever Value']))
            fval.append(state['Current Best Value'])

            for i in range(numdim):
                fvalXvec[i].append(state['Current Best Sample'][i])
                meanXvec[i].append(state['Sample Means'][i])
                width[i].append(state['Max Distances'][i])

            if (live == False):
                continue
            
            if ( not plt.fignum_exists(fig.number) ):
                print("[Korali] Figure closed - Bye!")
                exit(0)

            draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, fvalXvec, meanXvec, width, colors, names, live)
            pauseLight(0.05)

    if live == False: 
        draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, fvalXvec, meanXvec, width, colors, names, live)
        fig.show()

    plt.show()
    print("[Korali] Figure closed - Bye!")


# Create Plot from Data
def draw_figure(fig, ax, src, idx, numeval, numdim, fval, dfval, fvalXvec, meanXvec, width, colors, names, live):
    #fig, ax = plt.subplots(2,2,num='DEA live diagnostics: {0}'.format(src),figsize=(8,8))

    plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)), fontweight='bold', fontsize=12 )

    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    drawMulticoloredLine(ax[0,0], numeval, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(numeval, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    if ( (idx == 2) or (live == False) ):
        ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

    # Upper Right Plot
    ax[0,1].set_title('Objective Variables')
    ax[0,1].grid(True)
    for i in range(numdim):
        ax[0,1].plot(numeval, fvalXvec[i], color = colors[i], label=names[i])
    if ( (idx == 2) or (live == False) ):
        ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

    # Lower Right Plot
    ax[1,0].set_title('Width Population')
    ax[1,0].grid(True)
    for i in range(numdim):
        ax[1,0].plot(numeval, width[i], color = colors[i])

    # Lower Left Plot
    ax[1,1].set_title('Mean Population')
    ax[1,1].grid(True)
    for i in range(numdim):
        ax[1,1].plot(numeval, meanXvec[i], color = colors[i], label=names[i])
    if ( (idx == 2) or (live == False) ):
        ax[1,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)




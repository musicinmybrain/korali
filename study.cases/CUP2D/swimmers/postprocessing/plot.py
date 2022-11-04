#! /usr/bin/env python3

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import korali

import colorsys
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(1, amount * c[1]), c[2])

import glob

def loadTestingReturns( prefix, path ):
    # Go to target directory and get list with wished files
    currentDir = os.getcwd()
    os.chdir(path)
    files = sorted(glob.glob("{}*".format(prefix)))

    # Iterate over found files and extract testing returns
    returns = []
    for file in files:
        current = []
        with open(file, 'r') as read_obj:
            for line in read_obj:
                if "[Korali]    + (Average) Cumulative Reward:" in line:
                    current.append(float(line[-10:]))
        returns.append(current)

    # go back to original dir
    os.chdir(currentDir)

    # return matrix
    return np.array(returns)

def plotTestReturn( axs ):
    radiusHalfdisk = 0.03 + np.arange(0,10)*(0.07-0.03)/9.0
    returnsHalfDisk = loadTestingReturns( "halfDisk_largeDomain.eval_out_", "/project/s929/pweber/karmanPaper/runs/halfDisk_largeDomain.eval")
    returnsHalfDiskHydrofoil = loadTestingReturns( "halfDisk_largeDomain.eval.hydrofoil_out_", "/project/s929/pweber/karmanPaper/runs/halfDisk_largeDomain.eval.hydrofoil")

    frequencyHydrofoil = 0.28 + np.arange(0,10)*(1.37-0.28)/9.0
    returnsHydrofoil = loadTestingReturns( "hydrofoil_largeDomain.eval_out_", "/project/s929/pweber/karmanPaper/runs/hydrofoil_largeDomain.eval")
    returnsHydrofoilHalfdisk = loadTestingReturns( "hydrofoil_largeDomain.eval.halfdisk_out_", "/project/s929/pweber/karmanPaper/runs/hydrofoil_largeDomain.eval.halfdisk")

    returnsMultitaskHalfdisk = loadTestingReturns( "multitask_largeDomain.eval.halfdisk_out_", "/project/s929/pweber/karmanPaper/runs/multitask_largeDomain.eval.halfdisk")
    returnsMultitaskHydrofoil = loadTestingReturns( "multitask_largeDomain.eval_out_", "/project/s929/pweber/karmanPaper/runs/multitask_largeDomain.hydrofoil.eval")

    axs[0].plot(radiusHalfdisk, np.mean(returnsHalfDisk,axis=1), color="C0", linewidth=2)
    axs[0].fill_between(radiusHalfdisk, returnsHalfDisk.min(axis=1), returnsHalfDisk.max(axis=1), color="C0", alpha=0.5)

    axs[0].plot(radiusHalfdisk, np.mean(returnsMultitaskHalfdisk, axis=1), color="C2")
    axs[0].fill_between(radiusHalfdisk, returnsMultitaskHalfdisk.min(axis=1), returnsMultitaskHalfdisk.max(axis=1), color="C2", alpha=0.5)

    axs[0].plot(radiusHalfdisk, np.mean(returnsHydrofoilHalfdisk,axis=1), color="C1", linewidth=2)
    axs[0].fill_between(radiusHalfdisk, returnsHydrofoilHalfdisk.min(axis=1), returnsHydrofoilHalfdisk.max(axis=1), color="C1", alpha=0.5)

    axs[0].set_xlabel('Radius')
    axs[0].set_ylabel('Testing Return')
    axs[0].grid(True, which='minor')

    axs[1].plot(frequencyHydrofoil, np.mean(returnsHydrofoil,axis=1), color="C1", linewidth=2)
    axs[1].fill_between(frequencyHydrofoil, returnsHydrofoil.min(axis=1), returnsHydrofoil.max(axis=1), color="C1", alpha=0.5)

    axs[1].plot(frequencyHydrofoil, np.mean(returnsMultitaskHydrofoil, axis=1), color="C2", linewidth=2)
    axs[1].fill_between(frequencyHydrofoil, returnsMultitaskHydrofoil.min(axis=1), returnsMultitaskHydrofoil.max(axis=1), color="C2", alpha=0.5)

    axs[1].plot(frequencyHydrofoil, np.mean(returnsHalfDiskHydrofoil, axis=1), color="C0", linewidth=2)
    axs[1].fill_between(frequencyHydrofoil, returnsHalfDiskHydrofoil.min(axis=1), returnsHalfDiskHydrofoil.max(axis=1), color="C0", alpha=0.5)

    axs[1].set_xlabel('Frequency')
    axs[1].grid(True, which='minor')

def parseRM( dir ):

    # Load from Folder containing Results
    configFile = dir + '/latest'
    if (not os.path.isfile(configFile)):
        print("[Korali] Error: Did not find any results in the {0}...".format(configFile))
        exit(-1)

    with open(configFile) as f:
        config = json.load(f)

    dataFile = dir + '/state.json'
    if (not os.path.isfile(dataFile)):
        print("[Korali] Error: Did not find any results in the {0}...".format(dataFile))
        exit(-1)

    with open(dataFile) as f:
        data = json.load(f)

    return data, config

def parseActions( dir, numTests ):

    # Load from Folder containing Results
    data = []
    for s in range(numTests):
        actionFile = str(dir) + "/../_testingResults/sample{:03d}/actions0.txt".format(s)
        if (not os.path.isfile(actionFile)):
            print("[Korali] Error: Did not find any results in {0}...".format(actionFile))
            exit(-1)

        with open(actionFile) as f:
            data.append(np.loadtxt(actionFile))

    return data

def environment( data, config, xlim, ylim, N, s ):
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)

    averageValue = np.zeros((N,int(aspect*N)+1))
    averagedPolicyParameter = np.zeros((N,int(aspect*N)+1))

    # Containers to store state values and policy parameters
    states = []
    stateValues = []
    policyParams  = []
    replayMemory = data["Experience Replay"]
    for experience in replayMemory:
        state = np.array(experience["State"]).reshape((-1,)).tolist()
        states.append(state)
        s["State"]  = state
        s["Reward"] = 0

        # Getting new action
        s.update()

        # Append state value and distribution parameters
        stateValues.append(s["Policy"]["State Value"][0])
        policyParams.append(s["Policy"]["Distribution Parameters"][0])

    s["State"]  = state
    s["Reward"] = 0
    s["Termination"] = "Terminal"

    # need numpy array for the following
    states = np.array(states)
    stateValues = np.array(stateValues)
    policyParams  = np.array(policyParams)

    print("states:", states)
    print("stateValues", stateValues)
    print("policyParams", policyParams)

    # Get state rescaling factor and undo scaling
    rescalingMeans = np.array(config["Solver"]["State Rescaling"]["Means"])
    rescalingSigmas = np.array(config["Solver"]["State Rescaling"]["Sigmas"])

    rescalingMeans = np.reshape(rescalingMeans,(-1,))
    rescalingSigmas = np.reshape(rescalingSigmas,(-1,))

    stateX = 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
    stateY = 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

    for i in range(len(xCoords)-1):
        for j in range(len(yCoords)-1):
            indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                    & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
            averageValue[i,j] = np.mean(stateValues[indices])
    np.savetxt("stateValues.txt", averageValue)

    filenames = [ [ "meanAction1", "meanAction2"], [ "stdAction1", "stdAction2"] ]
    for l in range(2): #mean/sigma
        for k in range(2): #components
            for i in range(len(xCoords)-1):
                for j in range(len(yCoords)-1):
                        indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                                & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
                        averagedPolicyParameter[i,j] = np.mean(policyParams[indices,l*2+k])
            np.savetxt(filenames[l][k]+".txt", averagedPolicyParameter)

def evaluateNeuralNetwork( data, config, xlim, ylim, N, path ):
    # Create Korali Engine and Experiment
    k = korali.Engine()
    e = korali.Experiment()

    # Load Experiment from file
    found = e.loadState("_trainingResults/latest")
    if found == True: 
        print("[Korali] Evaluation results found...\n")
    else:
        print(stderr, "[Korali] Error: cannot find previous results\n")
        exit(0)

    # Configure Korali
    e["Problem"]["Environment Function"] = lambda x : environment( data, config, xlim, ylim, N, x );
    e["File Output"]["Path"] = "_trainingResults";
    e["Solver"]["Mode"] = "Testing";

    # random seeds for environment
    e["Solver"]["Testing"]["Sample Ids"] = [ 42 ];

    # launch testing
    k.run(e);

def plotValueFunctionRM( ax, data, config, xlim, ylim, N ):
    # Binning of x- and y- coordines
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)
    averageValue = np.empty((N,int(aspect*N)+1))
    averageValue.fill(np.nan)

    # Fill state and state-value vector with data from replay memory
    states = []
    stateValues = []
    replayMemory = data["Experience Replay"]
    for experience in replayMemory:
        states.append(experience["State"])
        stateValues.append(experience["State Value"])
    states = np.array(states)
    stateValues = np.array(stateValues)

    # Get state rescaling factor and undo scaling
    rescalingMeans = np.array(config["Solver"]["State Rescaling"]["Means"])
    rescalingSigmas = np.array(config["Solver"]["State Rescaling"]["Sigmas"])

    rescalingMeans = np.reshape(rescalingMeans,(-1,))
    rescalingSigmas = np.reshape(rescalingSigmas,(-1,))

    states = np.reshape(states, (-1,states.shape[2]))
    stateX = 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
    stateY = 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

    for i in range(len(xCoords)-1):
        for j in range(len(yCoords)-1):
            indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                    & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
            averageValue[i,j] = np.mean(stateValues[indices])

    X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

    im = ax.pcolormesh(X, Y, averageValue, cmap='viridis')
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    fig.colorbar(im, ax=ax)

    # Add circle
    circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
    ax.add_patch(circle)

def plotPolicyRM( ax, data, config, xlim, ylim, N ):
    # Binning of x- and y- coordines
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)
    averagedPolicyParameter = np.zeros((N,int(aspect*N)+1))
    averagedPolicyParameter.fill(np.nan)

    # Fill state and state-value vector with data from replay memory
    states = []
    policyParams  = []
    replayMemory = data["Experience Replay"]
    for experience in replayMemory:
        states.append(experience["State"])
        policyParams.append(experience["Current Policy"]["Distribution Parameters"])
    states = np.array(states)
    policyParams  = np.array(policyParams)

    policyParams = np.reshape(policyParams,(-1,policyParams.shape[2]))

    # Get state rescaling factor and undo scaling
    rescalingMeans = np.array(config["Solver"]["State Rescaling"]["Means"])
    rescalingSigmas = np.array(config["Solver"]["State Rescaling"]["Sigmas"])

    rescalingMeans = np.reshape(rescalingMeans,(-1,))
    rescalingSigmas = np.reshape(rescalingSigmas,(-1,))

    states = np.reshape(states, (-1,states.shape[2]))
    stateX = 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
    stateY = 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

    for l in range(2): #mean/sigma
        for k in range(2): #components
            for i in range(len(xCoords)-1):
                for j in range(len(yCoords)-1):
                        indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                                & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
                        averagedPolicyParameter[i,j] = np.mean(policyParams[indices,l*2+k])

            X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

            axs[l][k].set_xlim([0, xlim])
            axs[l][k].set_ylim([0, ylim])
            axs[l][k].set_xlabel('x')
            axs[l][k].set_ylabel('y')
            axs[l][k].grid(True, which='minor')
            if l == 0 and k == 0:
                im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu', vmin=-1, vmax=1)
            elif l == 0 and k == 1:
                im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu')
            else:
                im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='viridis')
            
            fig.colorbar(im, ax=ax[l][k])

            # Add circle
            circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
            axs[l][k].add_patch(circle)

def plotValue( averageValue, xlim, ylim, N ):
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)

    X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

    im = ax.pcolormesh(X, Y, averageValue, cmap='viridis')
    fig.colorbar(im, ax=ax)

    # Add circle
    circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
    ax.add_patch(circle) 

def plotPolicy( l, k, averagedPolicyParameter, xlim, ylim, N ):
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)

    X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

    if l == 0 and k == 0:
        im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu', vmin=-1, vmax=1)
    elif l == 0 and k == 1:
        im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu')
    else:
        im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='viridis')
    
    fig.colorbar(im, ax=ax[l][k])

    # Add circle
    circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
    axs[l][k].add_patch(circle)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='rl.statistics',
        description='Plot the statistics of the replay memory.')
    parser.add_argument(
        '--dir',
        help='Path to result files, separated by space',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--output',
        help='Indicates the output file path. If not specified, it prints to screen.',
        required=False)
    parser.add_argument(
        '--xlim',
        help='maximal extent in x-direction.',
        type = float,
        default = 2,
        required=False)
    parser.add_argument(
        '--ylim',
        help='maximal extent in y-direction.',
        type = float,
        default = 1,
        required=False)
    parser.add_argument(
        '--N',
        help='Number of points in x-direction.',
        type = int,
        default = 100,
        required=False)
    parser.add_argument(
        '--type',
        help='type of plot to create.',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--plotTestingReturns',
        help='Option to show testing returns.',
        action='store_true',
        required=False)
    parser.add_argument(
        '--numTests',
        help='Number of test runs performed.',
        type = int,
        default=0,
        required=False)

    args = parser.parse_args()

    if args.plotTestingReturns:
        fig, axs = plt.subplots(1, 2, figsize=(6,3), dpi=100, sharey=True)

        plotTestReturn( axs )

    if args.numTests > 0:
        data = parseActions( args.dir, args.numTests )

        fig, ax = plt.subplots(1, 1, figsize=(12,6))

        for i, d in enumerate(data):
            ax.plot(d[:,0], d[:,1], color=lighten_color("C0",0.1+0.1*i), linewidth=2)
            ax.set_ylim([-1, 1])
            ax.set_xlabel('time t')
            ax.set_ylabel('action')


    if args.type != "":
        data, config = parseRM(args.dir)

    if args.type == "value":
        ### Creating figure
        fig, ax = plt.subplots(1, 1, figsize=(6,3))

        ax.set_xlim([0, args.xlim])
        ax.set_ylim([0, args.ylim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, which='minor')
        plotValueFunctionRM( ax, data, config, args.xlim, args.ylim, args.N )
    elif args.type == "policy":
        ### Creating figure
        fig, axs = plt.subplots(2, 2, figsize=(12,6))

        plotPolicyRM( axs, data, config, args.xlim, args.ylim, args.N )
    elif args.type == "forward":
        evaluateNeuralNetwork( data, config, args.xlim, args.ylim, args.N, args.dir )

        ### Creating figure
        fig, ax = plt.subplots(1, 1, figsize=(6,3))

        ax.set_xlim([0, args.xlim])
        ax.set_ylim([0, args.ylim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, which='minor')

        averageValue = np.loadtxt("stateValues.txt", args.xlim, args.ylim, args.N )
        plotValue( averageValue )

        ### Creating figure
        fig, axs = plt.subplots(2, 2, figsize=(12,6))

        filenames = [ [ "meanAction1", "meanAction2"], [ "stdAction1", "stdAction2"] ]
        for l in range(2): #mean/sigma
            for k in range(2): #components
                averagedPolicyParameter = np.loadtxt(filenames[l][k]+".txt")
                plotPolicy( l, k, averagedPolicyParameter, args.xlim, args.ylim, args.N )

    ### Show/save plot
    fig.tight_layout()
    if (args.output is None):
        plt.show()
    else:
        plt.savefig(args.output)
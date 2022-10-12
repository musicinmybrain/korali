import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
import argparse

import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

import matplotlib.colors as mc
import colorsys

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

def plotTrajectory( root, obstacleId, output ):
    data = np.loadtxt(root+"/velocity_{}.dat".format(obstacleId), skiprows=1)

    fig, ax = plt.subplots(1, 2,sharex=True,figsize=(6,3), dpi=100)

    ax[0].plot(data[:,0], data[:,2], color="C0")
    ax[0].plot(data[:,0], data[:,3], color="C1")
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("position")
    # plt.ylim([0.8,1.4])

    ax[1].plot(data[:,0], data[:,7], label='x-components', color="C0")
    ax[1].plot(data[:,0], data[:,8], label='y-components', color="C1")
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel("velocity")
    # plt.ylim([-1,1])

    index = data[:,0] > 10
    print("Average velocity:", np.mean(data[index,7]))

    ax[1].legend(bbox_to_anchor=(-0.5,-0.2, 2, 1), loc="lower left", frameon=False, ncol=2)
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

initialPositions = [
  [ 0.60, 0.90 ],
  [ 0.60, 1.10 ],
  [ 0.90, 0.80 ],
  [ 0.90, 1.00 ],
  [ 0.90, 1.20 ],
  [ 1.20, 0.90 ],
  [ 1.20, 1.10 ],
  [ 1.50, 0.80 ],
  [ 1.50, 1.00 ],
  [ 1.50, 1.20 ],
  [ 1.80, 0.90 ],
  [ 1.80, 1.10 ],
  [ 2.10, 0.80 ],
  [ 2.10, 1.00 ],
  [ 2.10, 1.20 ],
  [ 2.40, 0.90 ],
  [ 2.40, 1.10 ],
  [ 2.70, 0.80 ],
  [ 2.70, 1.00 ],
  [ 2.70, 1.20 ]
]

def plotTrajectories( root, nAgents, output ):
    roots = [ root, "/scratch/snx3000/pweber/CUP2D/20swimmers"]
    fig, ax = plt.subplots(1, 1,sharex=True,figsize=(12,3), dpi=100)
    for i, r in enumerate(roots):
        for obstacleId in range(nAgents):
            data = np.loadtxt(r+"/velocity_{}.dat".format(obstacleId), skiprows=1)
            if i == 0:
                ax.plot(data[:,2], data[:,3], "-k", zorder=5)
            else:
                ax.plot(data[:,2], data[:,3], "--", alpha=0.5)
            ax.set_xlabel("x-coordinate")
            ax.set_ylabel("y-coordinate")

    plt.ylim([0.7,1.3])
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def plotForces( root, obstacleId, output ):
    data = np.loadtxt(root+"/forceValues_{}.dat".format(obstacleId), skiprows=1)

    fig, ax = plt.subplots(1, 3,sharex=True,figsize=(12,3), dpi=100)

    # 1/2 * speed of obstacle^2 * fish length
    nonDimFactor = 0.15**2/2*0.2

    smoothnedThrust = savgol_filter(data[:,10], 101, 3)
    smoothnedDrag   = savgol_filter(data[:,11], 101, 3)
    smoothnedForce  = savgol_filter(data[:,1 ], 101, 3)

    ax[0].plot(data[:,0], smoothnedThrust/nonDimFactor, color="C0")
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("thrust $T$")
    ax[1].plot(data[:,0], -smoothnedDrag/nonDimFactor, color="C1")
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel("drag $D$")
    ax[2].plot(data[:,0], -smoothnedForce/nonDimFactor, color="C2")
    ax[2].set_xlabel("time $t$")
    ax[2].set_ylabel("Force $F$")
    # twinAx.plot(data[:,0], data[:,7], label='torque', color="C2")

    # plt.legend()
    ax[1].set_title(root)
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def compareForces( root, output ):
    cases = ["/inPhase_", "/offPhase_"]
    fig, ax = plt.subplots(1, 1,sharex=True,figsize=(6,4), dpi=100)
    cmaps = [ matplotlib.cm.get_cmap('Blues'), matplotlib.cm.get_cmap('Greens') ]
    for c, case in enumerate(cases):
        cmap = cmaps[c]
        colCurrIndex = 0.0
        for j in range(1,10):
            averageForces = []
            lateralDisplacement = []
            for i in range(2,21):
                colorIndx = j/10
                print("Reading"+root+case+"{}_{}".format(j,i)+"...")
                data0 = np.loadtxt(root+case+"{}_{}".format(j,i)+"/forceValues_0.dat".format(0), skiprows=1)
                data1 = np.loadtxt(root+case+"{}_{}".format(j,i)+"/forceValues_1.dat".format(1), skiprows=1)

                velData0 = np.loadtxt(root+case+"{}_{}".format(j,i)+"/forceValues_0.dat".format(0), skiprows=1)

                # 1/2 * (fish length / swimmin period)^2 * fish length
                speed = 0.02*j
                nonDimFactor = 0.2*speed**2/2
                index0 = data0[:,0] > 5
                index1 = data1[:,0] > 5
                averageForce = (np.mean(data0[index0,1]) + np.mean(data1[index1,1])) / (2*nonDimFactor)
                averageForces.append(averageForce)
                lateralDisplacement.append(2*i/10.0)
            
            colorIndx = j/10
            ax.plot( lateralDisplacement, averageForces, color=cmap(colorIndx) )

    ax.set_xlabel("lateral displacement $\Delta y/L$")
    ax.set_ylabel("force coeffiecient")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def plotEnergy( root, obstacleId, output ):
    data = np.loadtxt(root+"/powerValues_{}.dat".format(obstacleId), skiprows=1)

    fig, ax = plt.subplots(1, 3, sharex=True,figsize=(12,3), dpi=100)

    # 1/2 * speed of obstacle^2 * fish length^2
    nonDimFactor = 0.15**2/2*0.2**2

    smoothnedThrustPower      = savgol_filter(data[:,1], 101, 3)
    smoothnedDragPower        = savgol_filter(data[:,2], 101, 3)
    smoothnedDeformationPower = savgol_filter(data[:,7], 101, 3)

    ax[0].plot(data[:,0], smoothnedThrustPower/nonDimFactor, color="C0")
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("thrust power $P_T$")
    ax[1].plot(data[:,0], smoothnedDragPower/nonDimFactor, color="C1")
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel("drag power $P_D$")
    ax[2].plot(data[:,0], smoothnedDeformationPower/nonDimFactor, color="C2")
    ax[2].set_xlabel("time $t$")
    ax[2].set_ylabel("deformation power $P_S$")
    plt.tight_layout()
    plt.show()

    smoothnedEfficiency = savgol_filter(data[:,-1], 51, 3)

    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(6,3), dpi=100)
    ax.plot(data[:,0], smoothnedEfficiency, label='efficiency', color="C3") #or -1
    ax.set_xlabel("time $t$")
    ax.set_ylabel("efficiency $\eta$")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def compareDisplacement():
    y0 = 0.5
    output = "displacement.png"
    # output = ""
    # dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish.eval/MAcolumnFish_yDisplacementMinimization/_testingResults/sample0001", "/scratch/snx3000/pweber/korali/MAcolumnFish.eval/MAcolumnFish_efficiency/_testingResults/sample0001"]
    dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish_Displacement.eval/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_Efficiency.eval/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_Both.eval/_testingResults/sample0000"]

    fig, ax = plt.subplots(3, 1, sharex=True,figsize=(6,6), dpi=100)

    for j, root in enumerate(dirs):
        print(root)
        for obstacleId in range(0,4):
            data = np.loadtxt(root+"/velocity_{}.dat".format(obstacleId), skiprows=1)
            ax[j].plot(data[:,0], np.abs(data[:,3]-y0), color="C{}".format(obstacleId), label="agent {}".format(obstacleId), linewidth=2) #or -1
            mean = np.mean(np.abs(data[:,3]-y0))
            ax[j].axhline(mean, linestyle="--", color="C{}".format(obstacleId))
            print("mean displacement agent {}: ".format(obstacleId), mean)
    ax[2].set_xlabel("time $t$")
    ax[0].set_ylabel("$|\Delta y|$")
    ax[1].set_ylabel("$|\Delta y|$")
    ax[2].set_ylabel("$|\Delta y|$")
    ax[0].set_title("minimize displacement")
    ax[1].set_title("maximize efficiency")
    ax[2].set_title("both")
    # ax[0].legend()
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=4, facecolor="white", edgecolor="white")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

# def compareForces():
#     output = "forces.png"
#     dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish_efficiency/_testingResults/sample0001", "/scratch/snx3000/pweber/korali/MAcolumnFish_yDisplacementMinimization/_testingResults/sample0001"]

#     fig, ax = plt.subplots(1, 1, sharex=True,figsize=(6,3), dpi=100)

#     # 1/2 * speed of obstacle^2 * fish length
#     nonDimFactor = 0.15**2/2*0.2

#     for j, root in enumerate(dirs):
#         for obstacleId in range(0,4):
#             data = np.loadtxt(root+"/forceValues_{}.dat".format(obstacleId), skiprows=1)
#             ax.plot(data[:,0], -data[:,11]/nonDimFactor, color=lighten_color("C{}".format(j*2),obstacleId*0.5), linewidth=2) #or -1
#     ax.set_xlabel("time $t$")
#     ax.set_ylabel("lateral displacement $\Delta y$")
#     plt.tight_layout()
#     if output != "":
#         plt.savefig(output)
#     else:
#         plt.show()

def compareEfficiency():
    output = "efficiency.png"
    # output = ""
    # dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish.eval/MAcolumnFish_yDisplacementMinimization/_testingResults/sample0001", "/scratch/snx3000/pweber/korali/MAcolumnFish.eval/MAcolumnFish_efficiency/_testingResults/sample0001"]
    dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish_Displacement.eval/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_Efficiency.eval/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_Both.eval/_testingResults/sample0000"]

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6), dpi=100)

    for j, root in enumerate(dirs):
        print(root)
        for obstacleId in range(0,4):
            data = np.loadtxt(root+"/powerValues_{}.dat".format(obstacleId), skiprows=1)
            # smoothned = savgol_filter(data[:,-1], 51, 3)
            # averaged = uniform_filter1d(data[:,-1], size=5000)
            filterWidth = 10000
            # averaged = np.convolve(data[:,-1], np.ones(filterWidth)/filterWidth, mode='valid')
            averageReturns = np.cumsum(data[:,-1])
            averaged = (averageReturns[filterWidth:]-averageReturns[:-filterWidth])/float(filterWidth)
            averaged = np.insert(averaged, 0,averageReturns[filterWidth-1]/float(filterWidth))
            ax[j].plot(data[filterWidth-1:,0], averaged, color="C{}".format(obstacleId), label="agent {}".format(obstacleId), linewidth=2) #or -1
            mean = np.mean(data[:,-1])
            ax[j].axhline(mean, linestyle="--", color="C{}".format(obstacleId))
            print("mean efficiency agent {}: ".format(obstacleId), mean)
    ax[2].set_xlabel("time $t$")
    ax[0].set_ylabel("$|\eta|$")
    ax[1].set_ylabel("$|\eta|$")
    ax[2].set_ylabel("$|\eta|$")
    ax[0].set_title("minimize displacement")
    ax[1].set_title("maximize efficiency")
    ax[2].set_title("both")
    # ax[0].legend()
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=4, facecolor="white", edgecolor="white")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def plotEfficiency():
    rootControlled = "/scratch/snx3000/mchatzim/CUP2D/School_Control_09Fish/"
    rootUncontrolled = "/scratch/snx3000/mchatzim/CUP2D/School_NoControl_09Fish/"
    rootSingle = "/scratch/snx3000/mchatzim/CUP2D/single-fish/"

    row = 9

    powDataSingle = np.loadtxt(rootSingle+"powerValues_0.dat", skiprows=1)
    powDataSingleAverage = powDataSingle[:,row]
    powDataSingleSmoothned = savgol_filter(powDataSingleAverage, 51, 3)
    plt.plot(powDataSingle[:,0], powDataSingleSmoothned, linewidth=1, label='single fish')
    powDataSingleAveraged = uniform_filter1d(powDataSingleAverage, size=1000)
    plt.plot(powDataSingle[:,0], powDataSingleAveraged, "--", linewidth=2, label='single fish, average')

    numSwimmersUncontrolled = 9
    powDataUncontrolled = np.loadtxt(rootUncontrolled+"powerValues_0.dat", skiprows=1)
    powDataUncontrolledAverage = np.zeros(powDataUncontrolled[:,row].shape)
    for i in range(0,numSwimmersUncontrolled):
        powDataUncontrolled = np.loadtxt(rootUncontrolled+"powerValues_{}.dat".format(i), skiprows=1)
        powDataUncontrolledAverage = powDataUncontrolledAverage + powDataUncontrolled[:,row]
    powDataUncontrolledSmoothened = savgol_filter(powDataUncontrolledAverage, 51, 3)
    plt.plot(powDataUncontrolled[:,0], powDataUncontrolledSmoothened/numSwimmersUncontrolled, linewidth=1, label='9 fish, uncontrolled')
    powDataUncontrolledAveraged = uniform_filter1d(powDataUncontrolledAverage, size=1000)
    plt.plot(powDataUncontrolled[:,0], powDataUncontrolledAveraged/numSwimmersUncontrolled, "--", linewidth=2, label='9 fish, uncontrolled, average')

    powDataControlled = np.loadtxt(rootControlled+"powerValues_0.dat", skiprows=1)
    powDataAverage = np.zeros(powDataControlled[:,row].shape)
    for i in range(0,9):
        powDataControlled = np.loadtxt(rootControlled+"powerValues_{}.dat".format(i), skiprows=1)
        powDataAverage = powDataAverage + powDataControlled[:,row]
    powDataAverageSmoothened = savgol_filter(powDataAverage, 51, 3)
    plt.plot(powDataControlled[:,0], powDataAverageSmoothened/9, linewidth=1, label='9 fish, controlled')
    powDataAverageAveraged = uniform_filter1d(powDataAverage, size=1000)
    plt.plot(powDataControlled[:,0], powDataAverageAveraged/9, "--", linewidth=2, label='9 fish, controlled, average')

    plt.xlim([0,30])
    plt.legend()
    plt.show()

def animateCoM():
    path = "/scratch/snx3000/pweber/korali/hydrofoil_largeDomain.eval/_testingResults/sample000/"
    numFish = 1
    
    # Load data, set up figure, axis, and plot element we want to animate
    fig = plt.figure()
    fig.patch.set_alpha(0.)
    ax = plt.axes(xlim=(0.4, 2),ylim=( 0.6, 1.4 ) )
    ax.set_axis_off()
    lines = []
    data = []
    for i in range(numFish):
        data.append( np.loadtxt(path+"velocity_{}.dat".format(i), skiprows=1) )
        lines.append(ax.plot([], [], lw=2, color="C{}".format(i))[0])

    # initialization function: plot the background of each frame
    def init():
      for line in lines:
        line.set_data([], [])
      return lines

    # animation function.  This is called sequentially
    def animate(i):
      plt.title("Center of Mass, t={}".format(data[0][i,0]))
      for j in range(numFish):
        lines[j].set_data(data[j][:i,2], data[j][:i,3])
      return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data[0][:,0]), interval=1, blit=True)

    # save the animation as an mp4
    anim.save('CoM_animation_fish.mp4', fps=10)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='cup2D.dynamics',
        description='Plot the dynamical quantities of a CUP2D run.')
    parser.add_argument(
        '--dir',
        help='Path to result file',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--output',
        help='Name of saved figure.',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--obstacleId',
        help='Number of obstacle to plot.',
        type = int,
        default = 1,
        required=False)
    args = parser.parse_args()


    # plotTrajectory(args.dir, args.obstacleId, args.output)
    # plotTrajectories(args.dir, args.obstacleId, args.output)
    # plotForces(args.dir, args.obstacleId, args.output)
    # plotEnergy(args.dir, args.obstacleId, args.output)
    # compareDisplacement()
    compareForces(args.dir, args.output)
    # compareEfficiency()
    # animateCoM()
    # plotEfficiency()
import numpy as np
import os
folder = '/scratch/snx3000/anoca/CUP2D/OSP/'

amplitudes = np.linspace(-2, 2, 41)
frequencies = np.linspace(0, 2, 21)

# folder output name : Ax.x_fx.x

# constant parameters of simulation
bpdx=3
bpdy=4
levels=4
rtol=0.1
ctol=0.01
extent=0.7
cfl=0.4
xpos=0.1
ypos1=0.25
ypos2=0.45
xvel=0.15
maaxis=0.0405
miaxis=0.0135
nu=0.000243

bot=2
freq1=0.5

# range of the ratios

# r_a : bot / top => [-1, 1]
# r_f : freq2 / freq1 => [0,4]


tests = ['test1/', 'test2/']
prefix = '[@ ' + folder
suffix = ' @] '

counter = 0

with open('task.txt', 'w') as f:

    for a in amplitudes:
        for fr in frequencies:
            output = "A" + str(round(a, 1)) + "_f" + str(round(fr, 1)) + "/"
            OPTIONS= f"-bpdx {bpdx} -bpdy {bpdy} -levelMax {levels} -Rtol {rtol} -Ctol {ctol} -extent {extent} -CFL {cfl} -tdump 0.5 -nu {nu} -poissonTol 1.0e-3 -tend 60 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver cuda_iterative"
            ## two WINDMILLS, constant angular velocity of 4.0hz
            OBJECTS= f"windmill semiAxisX={maaxis} semiAxisY={miaxis} xpos={xpos} ypos={ypos1} bForced=1 bFixed=1 xvel={xvel} tAccel=0 bBlockAng=1 angvelmax={bot} freq={freq1}, windmill semiAxisX={maaxis} semiAxisY={miaxis} xpos={xpos} ypos={ypos2} bForced=1 xvel={xvel} tAccel=0 bBlockAng=1 angvelmax={round(a, 1)} freq={round(fr,1)}"

            f.write(prefix + output + suffix + './simulation ' + OPTIONS + ' -shapes ' + OBJECTS + '\n')

            # create all the folders
            os.mkdir(folder + output)
            # cp all the ./simulation executables to the folders
            os.system("cp ~/korali/examples/study.cases/CUP2D/_deps/CUP-2D/makefiles/simulation " + folder + output)
            counter += 1
            print(counter)







#with open('launchscript.txt', 'w') as f:

    #f.write('readme')
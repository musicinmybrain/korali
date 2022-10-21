# this script is used to create a task file, for launching many CUP2D windmills simulations
# it creates simulation folders in the folder specified hereunder, putting the ./simulation executable in those folders
# the parameters of the simulation

import numpy as np
import os
folder = '/scratch/snx3000/anoca/CUP2D/cmaes/'

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

# obtain the optimal parameters from the data. need to run get_data_cmaes.py in order to get these values.
fol = '../data/'
npzfiles = np.load(fol + 'cmaes_data.npz')
parameters = npzfiles['optimum'] # 10 x 4 array of the optimum parameters
num_sims = parameters.shape[0]


prefix = '[@ ' + folder
suffix = ' @] '

counter = 0

with open('task.txt', 'w') as f:
    for i in range(num_sims):
        output = str(i+1) + "/"
        a1, a2, f1, f2 = parameters[i]
        OPTIONS= f"-bpdx {bpdx} -bpdy {bpdy} -levelMax {levels} -Rtol {rtol} -Ctol {ctol} -extent {extent} -CFL {cfl} -tdump 0.0 -nu {nu} -poissonTol 1.0e-3 -tend 60 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver cuda_iterative"
        ## two WINDMILLS, constant angular velocity of 4.0hz
        OBJECTS= f"windmill semiAxisX={maaxis} semiAxisY={miaxis} xpos={xpos} ypos={ypos1} bForced=1 bFixed=1 xvel={xvel} tAccel=0 bBlockAng=1 angvelmax={a1} freq={f1}, windmill semiAxisX={maaxis} semiAxisY={miaxis} xpos={xpos} ypos={ypos2} bForced=1 xvel={xvel} tAccel=0 bBlockAng=1 angvelmax={a2} freq={f2}"

        f.write(prefix + output + suffix + './simulation ' + OPTIONS + ' -shapes ' + OBJECTS + '\n')

        # create all the folders
        os.mkdir(folder + output)
        # cp all the ./simulation executables to the folders
        os.system("cp ~/korali/examples/study.cases/CUP2D/_deps/CUP-2D/makefiles/simulation " + folder + output)
        
        counter += 1
        print(counter)
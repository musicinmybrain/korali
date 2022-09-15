import numpy as np
import os
folder = ''

amplitudes = np.linspace(-2, 2, 41)
frequencies = np.linspace(0, 2, 21)

# folder output name : Ax.x_fx.x

top=2
freq1=0.5

prefix = '[@ ' + folder
suffix = ' @] '

counter = 0

with open('task.txt', 'w') as f:

    for a in amplitudes:
        for fr in frequencies:
            output = "A" + str(round(a, 1)) + "_f" + str(round(fr, 1))
            
            f.write('./convert ' + output + '\n')

        counter += 1
        print(counter)







#with open('launchscript.txt', 'w') as f:

    #f.write('readme')
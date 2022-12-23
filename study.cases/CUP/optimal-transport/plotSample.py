
import numpy as np
import matplotlib.pyplot as plt

def logDivision(start, end, nvertices):
    vertices = np.zeros(nvertices)
    for idx in range(nvertices):
        vertices[idx] = np.exp(idx / (nvertices - 1.0) * np.log(end-start+1.0)) - 1.0 + start

    return vertices

sample1 = [0.0029774214492784834, 0.0012545252423366115, 0.0022471337578267294, 0.0004487711837245371]
sample2 = [0.004315617345056653, 0.002357595158866241, 0.00045192343087713835, 0.003751587911891705, 0.00029270425844050757, 0.001791944628598642, 0.0006044968601873586, 0.0005902863745657632]

#sample1 = [0.009899837203413772, 0.006042697786444769, 0.007737424752668662, 0.004035855841072438]
#sample2 = [0.009963723478185877, 0.009494772171610993, 0.007958669389234783, 0.004936448245926574, 0.005157520884244215, 0.007214823809055573, 0.004063898057291126, 0.0060274437185140455]

sample1 = [0.0] + sample1
sample2 = [0.0] + sample2

vertices1 = logDivision(0.2, 0.8, len(sample1))
vertices2 = logDivision(0.2, 0.8, len(sample2))

print(vertices1)

plt.title('Energy 0.001')
#plt.title('Energy 0.004')

plt.step(vertices1, sample1, label='D = 4 (T = 6.09)')
plt.step(vertices2, sample2, label='D = 8 (T = 6.00)')
#plt.step(vertices1, sample1, label='D = 4 (T = 3.07)')
#plt.step(vertices2, sample2, label='D = 8 (T = 3.11)')

plt.legend()
plt.xlabel('Location')
plt.ylabel('Force')
plt.ylim(0.0, 0.012)
plt.savefig('params.png')





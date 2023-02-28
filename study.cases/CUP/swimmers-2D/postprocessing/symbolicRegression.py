import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor

xlim = 1.0 # 10
ylim = 0.5 # 5 
N = 100
path = "/project/s1160/pweber/halfDisk_largeDomain"

# Load from Folder containing Results
configFile = path + '/latest'
if (not os.path.isfile(configFile)):
    print("[Korali] Error: Did not find any results in the {0}...".format(configFile))
    exit(-1)

with open(configFile) as f:
    config = json.load(f)

dataFile = path + '/state.json'
if (not os.path.isfile(dataFile)):
    print("[Korali] Error: Did not find any results in the {0}...".format(dataFile))
    exit(-1)

with open(dataFile) as f:
    data = json.load(f)

# Create coordinate tuples
aspect = ylim / xlim
# xCoords = np.linspace(0,xlim,N)
# yCoords = np.linspace(0,ylim,int(aspect*N)+1)
xCoords = np.linspace(-xlim,xlim,N)
yCoords = np.linspace(-ylim,ylim,int(aspect*N)+1)

coords = []
for i in range(len(xCoords)-1):
    for j in range(len(yCoords)-1):
        coords.append([xCoords[i],yCoords[j]])
coords = np.array(coords)

print(coords.shape)

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
# stateX = ( 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2 ) / 0.2
# stateY = ( 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2 ) / 0.2

stateX = ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
stateY = ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

# empty vector for policy parameters
averagedPolicyParameter = []
nanIndices = []
for i in range(len(xCoords)-1):
    for j in range(len(yCoords)-1):
            indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                    & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
            averagePolicyParameter = np.mean(policyParams[indices,1])
            if np.isnan(averagePolicyParameter):
                nanIndices.append(False)
            else:
                nanIndices.append(True)
                averagedPolicyParameter.append(averagePolicyParameter)
averagedPolicyParameter = np.array(averagedPolicyParameter)
coords = coords[nanIndices,:]

# Configure symbolic regression
model = PySRRegressor(
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

# Perform symbolic regression
# model.fit(coords, averagedPolicyParameter)
model = PySRRegressor.from_file("hall_of_fame_2023-01-11_104515.769.pkl")

# check results
print(model)
print(model.equations_)

# forward model
Z = model.predict(coords)

# plot results
fig, axs = plt.subplots(2, 1, figsize=(6,6))
# axs[0].tricontourf(coords[:,0], coords[:,1], averagedPolicyParameter, cmap='RdBu', vmin=-1, vmax=1)
contour = axs[0].tricontourf(coords[:,0], coords[:,1], averagedPolicyParameter, cmap='RdBu')
axs[0].set_title("data")
# axs[1].tricontourf(coords[:,0], coords[:,1], Z, cmap='RdBu', vmin=-1, vmax=1)
axs[1].tricontourf(coords[:,0], coords[:,1], Z, cmap='RdBu')
# axs[1].set_title("fit = 3.35 / x $\cdot$ sin(y + 0.60)")
# axs[1].set_title("fit = y $\cdot$ (x - 2.69)")
axs[0].set_ylabel('y')
axs[1].set_ylabel('y')
axs[1].set_xlabel('x')

# plt.colorbar(mappable = contour)
plt.tight_layout()
plt.savefig("symbolicRegression.eps")

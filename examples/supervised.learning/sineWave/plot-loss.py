#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt

file = "_korali_result"

if (not os.path.isfile(file+"/latest")):
	print("[Korali] Error: Did not find any results in the {0} folder...".format(file+"/latest"))
	exit(-1)

with open(file+"/latest") as f:
	data = json.load(f)

plt.plot(data["Solver"]["Loss History"])
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("loss.png")
import os
import json
import matplotlib.pyplot as plt

files = [ "old", "new" ]

for file in files:
	if (not os.path.isfile(file+"/latest")):
		print("[Korali] Error: Did not find any results in the {0} folder...".format(file+"/latest"))
		exit(-1)

	with open(file+"/latest") as f:
		data = json.load(f)

	if file == "all-in-one":
		plt.plot(data["Solver"]["Loss History"], label=file, color="k", linewidth=3)
	elif file == "new":
		plt.plot(data["Solver"]["Loss History"], label=file)
	else:
		plt.plot(data["Solver"]["Loss History"], label=file)

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("test.png")
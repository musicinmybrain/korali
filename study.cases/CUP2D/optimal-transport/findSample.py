import argparse
import json
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory to check for latest file.', required=True)
    parser.add_argument('--objective', type=int, help='Objective function.', required=True)
    parser.add_argument('--value', type=float, help='Value to look for.', required=True)

    args = parser.parse_args()

    filename = args.directory + '/latest'
    mindist = math.inf
    sample = []
    fsample = []
    with open(filename) as json_file:
        data = json.load(json_file)
        samplevalues = data["Solver"]["Sample Value Collection"]
        samples = data["Solver"]["Sample Collection"]


        sampleidx = -1
        objidx = args.objective
        target = args.value
        for idx, values in enumerate(samplevalues):
            dist = abs(values[objidx] - args.value)
            if dist < mindist:
                mindist = dist
                sampleidx = idx



        sample = samples[sampleidx]
        fsample = samplevalues[sampleidx]


    print("Sample Found: {}".format(sampleidx))
    print("Params: {}".format(sample))
    print("Objectives: {}".format(fsample))
    print("Dist: {}".format(mindist))




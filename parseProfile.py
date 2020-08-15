#!/usr/bin/python3

import sys
import json
import jsonpickle
import re
from simulator import Profile

# Converts Pipedream's profile to 
def main():
# node1 -- Input0 -- forward_compute_time=0.000, backward_compute_time=0.000, activation_size=0.0, parameter_size=0.000    
    profile = Profile()
    f = open(sys.argv[1])
    for line in f:
        components = line.split(' -- ')
        layerIdMatch = re.match('node(\d+)', components[0])
        if not layerIdMatch:
            break
        layerId = int(layerIdMatch.group(1))
        name = components[1]
        matchTimes = re.match('forward_compute_time=(\d*\.\d+|\d+), backward_compute_time=(\d*\.\d+|\d+), activation_size=(\d*\.\d+|\d+), parameter_size=(\d*\.\d+|\d+)', components[2])
        fowardComp = float(matchTimes.group(1))
        backwardComp = float(matchTimes.group(2))
        activationSizeComp = float(matchTimes.group(3))
        parameterSize = float(matchTimes.group(4))
        # print("%d %s %f %f %f %f" % (layerId, name, fowardComp, backwardComp, activationSizeComp, parameterSize))
        # def addDatapoint(self, layerId, localBatch, computeTime, alreadySorted = False):
        batchSize = 64
        profile.addDatapoint(layerId, batchSize, [fowardComp, backwardComp])

    print(json.dumps(profile.datapoint))
    # print(jsonpickle.encode(profile, unpicklable=False))

if __name__ == "__main__":
    main()
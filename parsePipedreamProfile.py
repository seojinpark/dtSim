#!/usr/bin/python3

import sys
import json
import jsonpickle
import re
import glob
from os import listdir
from os.path import isfile, join
from profile import Profile
from trainingPlanEditor import Layer

# Converts Pipedream's profile to dtSim's profile.
def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments!")
        print("Usage:")
        print("python3 parseProfile.py \"path/to/folder/containing/txt/files/\"")
        return

    folderpath = sys.argv[1]
    if folderpath[-1] != '/':
        folderpath += '/'

    profile = Profile()
    
    # print(glob.glob(path + "*.txt"))
    onlyfiles = [f for f in listdir(folderpath) if isfile(join(folderpath, f))]
    
    isFirstFile = True
    trainingPlan = []
    layerInfo = {} # [layerId] = (layerId, name, parameterSize, activationSize)
    for filename in onlyfiles:
        batchSizeMatch = re.match('(\d+).txt', filename)
        if not batchSizeMatch:
            continue
        batchSize = int(batchSizeMatch.group(1))
        print("filepath:%s, batchSize:%d"%(folderpath+filename, batchSize))
        f = open(folderpath + filename)
        
        for line in f:
            components = line.split(' -- ')
            layerIdMatch = re.match('node(\d+)', components[0])
            if layerIdMatch:
                layerId = int(layerIdMatch.group(1))
                name = components[1]
                matchTimes = re.match('forward_compute_time=(\d*\.\d+|\d+), backward_compute_time=(\d*\.\d+|\d+), activation_size=(\d*\.\d+|\d+), parameter_size=(\d*\.\d+|\d+)', components[2])
                fowardComp = float(matchTimes.group(1))
                backwardComp = float(matchTimes.group(2))
                activationSize = float(matchTimes.group(3))
                parameterSize = float(matchTimes.group(4))
                # print("%d %s %f %f %f %f" % (layerId, name, fowardComp, backwardComp, activationSizeComp, parameterSize))
                # def addDatapoint(self, layerId, localBatch, computeTime, alreadySorted = False):
                # batchSize = 64
                profile.addDatapoint(layerId, batchSize, [fowardComp, backwardComp])

                # populate for training plan.
                # plan.append(Layer(2, 10000, [{"LayerId": 1, "InputBytesPerSample": 100}],
                #                  [{"id": 3, "localBatch": 32}, {"id": 5, "localBatch": 32}]))
                if isFirstFile:
                    layerInfo[layerId] = (layerId, name, parameterSize, activationSize)
            elif isFirstFile:
                prevLayerIdMatch = re.match('\s+node(\d+)', components[0])
                currLayerIdMatch = re.match('node(\d+)\s*', components[1])
                if not prevLayerIdMatch:
                    break
                prevLayerId = int(prevLayerIdMatch.group(1))
                currLayerId = int(currLayerIdMatch.group(1))
                # WARNING. only support sequencial layers.
                
                if len(trainingPlan) == 0: # If prevLayer is input layer, it is added specially.
                    layerId, name, parameterSize, activationSize = layerInfo[prevLayerId]
                    assert(layerId == prevLayerId)
                    trainingPlan.append(Layer(layerId, name, parameterSize, []))
                    
                layerId, name, parameterSize, activationSize = layerInfo[currLayerId]
                assert(layerId == currLayerId)
                if len(trainingPlan) >= layerId:
                    print("Warning! non-sequencial connection. Correct inputBytes manually for layer %d. (Pipedream doesn't supply enough info.)" % layerId)
                    trainingPlan[layerId-1].prevLayers.append(
                        {"LayerId": prevLayerId, "InputBytesPerSample": 0}
                    )
                else:
                    trainingPlan.append(Layer(layerId, name, parameterSize, 
                            [{"LayerId": prevLayerId, "InputBytesPerSample": (activationSize / batchSize)}]))
                
        isFirstFile = False
        
    # print(json.dumps(profile.datapoint))
    with open(folderpath + "profile.json", 'w') as outfile:
        json.dump(profile.datapoint, outfile)
    with open(folderpath + "plan_unassigned.json", "w") as outfile:
        # outfile.write(jsonpickle.encode(trainingPlan, unpicklable=False))
        planInJson = jsonpickle.encode(trainingPlan, unpicklable=False)
        json.dump(json.loads(planInJson), outfile, indent=2, sort_keys=False)

if __name__ == "__main__":
    main()
#!/usr/bin/python3

import sys
import json
import re
import glob
from os import listdir
from os.path import isfile, join
from trainingPlanEditor import Layer

# Converts Pipedream's profile to dtSim's profile.
def main():
    if len(sys.argv) != 5:
        print("Wrong number of arguments!")
        print("Usage:")
        print("python3 parsePipedreamPlan.py \"path_to_optimizer_stdout_file\" \"path_to_optimizer_gpusFile\" \"path_to_unassigned_plan_in_json\" batchSize")
        return
        
    batchSize = int(sys.argv[4])
    plan = []
    
    with open(sys.argv[3]) as unassignedPlanfile:
        plan = json.load(unassignedPlanfile)

    with open(sys.argv[1]) as optimizerStdoutFile:
        parsingRegionBegan = False
        stageId = 0
        stageToRanks = {}
        nextRank = 1
        for line in optimizerStdoutFile:
            if line == "(Split start, split end) / compute time taken per stage / replication factor per stage:\n":
                parsingRegionBegan = True
                continue
            if not parsingRegionBegan:
                continue
            
            match = re.match('\((\d+), (\d+)\) (\d*\.\d+|\d+) (\d*\.\d+|\d+)', line)
            if not match:
                break
            
            start = int(match.group(1))
            end = int(match.group(2))
            computeTime = float(match.group(3))
            replicas = int(match.group(4))
            stageToRanks[stageId] = (nextRank, replicas)
            nextRank += replicas
            
            stageId += 1

    with open(sys.argv[2]) as optimizerGpusFile:
        for line in optimizerGpusFile:
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
            
                stageId = int(re.match('stage_id=(\d+)', components[3]).group(1))
                
                if int(plan[layerId-1]["layerId"]) != layerId:
                    print("Not matched! %d %d" % (int(plan[layerId-1]["layerId"]), layerId))
                assert(int(plan[layerId-1]["layerId"]) == int(layerId))
            
                startRank = stageToRanks[stageId][0]
                replicas = stageToRanks[stageId][1]
                plan[layerId-1]["assignedAccelerators"] = []
                for i in range(replicas):
                    rank = startRank + i
                    localBatch = int(batchSize * (i + 1) / replicas) - int(batchSize * i / replicas)
                    plan[layerId-1]["assignedAccelerators"].append( {'id': rank, 'localBatch': localBatch} )

    print(json.dumps(plan, indent=2, sort_keys=False))

if __name__ == "__main__":
    main()
#!/usr/bin/python3

# Copyright (c) 2020 MIT
# 
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import sys
import jsonpickle
import json
from networkEditor import Network
from networkEditor import Simulation
from networkEditor import buildHostAndGpuNetwork
from trainingPlanEditor import buildSimplePlan
from profile import Profile

DEBUG = True
def simulate(trainingPlan, network, profiles, useGuidForAcceleratorIds=False):
    sim = Simulation(network)
    layersById = [None] * (len(trainingPlan) + 1)
    for layer in trainingPlan:
        layersById[layer["layerId"]] = layer

    # Step1. Forward Pass
    computeTasksByLayer = [dict() for x in range(len(trainingPlan) + 1)] # [layerId][acceleratorId] = ComputeTask
    for layer in trainingPlan: # trainingPlan must be sorted in the DAG order.
        numReplicas = len(layer["assignedAccelerators"])
        totalBatchSize = sum([assign["localBatch"] for assign in layer["assignedAccelerators"]])
        samplesAssigned = 0
        
        for assignment in layer["assignedAccelerators"]:
            #    def scheduleCompute(self, acceleratorId, layerId, computeTime, prevXferTasks = []):
            # aid = assignment['id']
            if useGuidForAcceleratorIds:
                aid = assignment['id']
            else:
                aid = network.accelerators[assignment['id']-1].guid

            lid = layer["layerId"]
            computeTime = profiles[network.elements[aid].model].getCost(0, lid, assignment['localBatch'])
            prevXferTasks = []
            
            for prevLayerPtr in layer["prevLayers"]:
                pl = layersById[prevLayerPtr["LayerId"]]
                plsa = 0
                if DEBUG:
                    totalPrevBatchSize = sum([pla["localBatch"] for pla in pl["assignedAccelerators"]])
                    assert(totalBatchSize == totalPrevBatchSize)
                for pla in pl["assignedAccelerators"]:
                    if useGuidForAcceleratorIds:
                        plaid = pla['id']
                    else:
                        plaid = network.accelerators[pla['id']-1].guid

                    if (plsa < samplesAssigned + assignment['localBatch']) and \
                            (plsa + pla["localBatch"] > samplesAssigned): # sample ranges overlap.
                        # Generate xfer task.
                        left = max(plsa, samplesAssigned)
                        right = min(plsa + pla["localBatch"], samplesAssigned + assignment['localBatch'])
                        xferSamples = right - left # TODO: need a unit test to check this calucation is correct..
                        assert(xferSamples > 0)
                        xferBytes = xferSamples * prevLayerPtr["InputBytesPerSample"]
                        # def scheduleXfer(self, src, dst, xferBytes, prevComputeTask = None):
                        print("Scheduled xfer for %d samples from %d to %d" % (xferSamples, plaid, aid))
                        prevXferTasks.append(sim.scheduleXfer(plaid, aid, xferBytes,
                                             computeTasksByLayer[prevLayerPtr["LayerId"]][plaid]))
                    if plsa >= samplesAssigned + assignment['localBatch']:
                        break
                    plsa += pla["localBatch"]
                
            samplesAssigned += assignment['localBatch']
            computeTasksByLayer[lid][aid] = sim.scheduleCompute(aid, lid, computeTime, prevXferTasks)
        assert(samplesAssigned == totalBatchSize)
    # Run forward-pass
    sim.run()
    sim.plotNetwork()

def run_example1():
    net = buildHostAndGpuNetwork(2, 2, 10, 10)
    net.printAllPaths()
    # net.plotNetwork()
    # trainingPlan = json.loads(buildSimplePlan())
    trainingPlan = json.load(open("simplePlan.json"))
    print(json.dumps(trainingPlan, indent=2, sort_keys=False))
    prof_v100 = Profile("profile_pipedream/P100/profile.json")
    # prof_v100 = Profile()
    # prof_v100.addDatapoint(1, 32, [100, 100])
    # prof_v100.addDatapoint(1, 64, [164, 150])
    # prof_v100.addDatapoint(2, 32, [132, 110])
    profiles = {"V100": prof_v100}
    simulate(trainingPlan, net, profiles)
    
def main():
    if len(sys.argv) == 1:
        run_example1()
    elif len(sys.argv) == 3:
        net = buildHostAndGpuNetwork(2, 2, 10, 10)
        profile = Profile(sys.argv[1])
        profiles = {"V100": profile} # TODO: support heterogeneous GPUs
        with open(sys.argv[2]) as f:
            trainingPlan = json.load(f)
        simulate(trainingPlan, net, profiles, False)
    else:
        print("Wrong number of args! Usage:")
        print("./simulator <path_to_profile> <path_to_plan>")

if __name__ == "__main__":
    main()

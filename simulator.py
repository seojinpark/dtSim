import jsonpickle
import json
from networkEditor import Network
from networkEditor import Simulation
from networkEditor import buildHostAndGpuNetwork
from trainingPlanEditor import buildSimplePlan


class Profile:
    def __init__(self):
        self.datapoint = {} # [<dict> layerId] = [(localBatch, computeTime), ...]
        
    def addDatapoint(self, layerId, localBatch, computeTime, alreadySorted = False):
        if layerId not in self.datapoint:
            self.datapoint[layerId] = []
        self.datapoint[layerId].append((localBatch, computeTime))
        if not alreadySorted:
            self.datapoint[layerId].sort() # TODO: make it efficient when performance matters.
        
    def getCost(self, layerId, localBatch):
        batch_a = 0
        compTime_a = 0
        batch_b = 0
        compTime_b = 0
        
        for dpLocalBatch, dpComputeTime in self.datapoint[layerId]:
            if localBatch <= dpLocalBatch:
                batch_b = dpLocalBatch
                compTime_b = dpComputeTime
                break
            else:
                batch_a = dpLocalBatch
                compTime_a = dpComputeTime
        
        assert(batch_b > 0)
        
        return (localBatch - batch_a + 0.0) * (compTime_b - compTime_a + 0.0) / (batch_b - batch_a + 0.0) + compTime_a

DEBUG = True
def simulate(trainingPlan, network, profiles):
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
            aid = assignment['id']
            lid = layer["layerId"]
            computeTime = profiles[network.elements[aid].model].getCost(lid, assignment['localBatch'])
            prevXferTasks = []
            
            for prevLayerPtr in layer["prevLayers"]:
                pl = layersById[prevLayerPtr["LayerId"]]
                plsa = 0
                if DEBUG:
                    totalPrevBatchSize = sum([pla["localBatch"] for pla in pl["assignedAccelerators"]])
                    assert(totalBatchSize == totalPrevBatchSize)
                for pla in pl["assignedAccelerators"]:
                    if plsa >= samplesAssigned:
                        # Generate xfer task.
                        left = max(plsa, samplesAssigned)
                        right = min(plsa + pla["localBatch"], samplesAssigned + assignment['localBatch'])
                        xferSamples = right - left # TODO: need a unit test to check this calucation is correct..
                        xferBytes = xferSamples * prevLayerPtr["InputBytesPerSample"]
                        # def scheduleXfer(self, src, dst, xferBytes, prevComputeTask = None):
                        prevXferTasks.append(sim.scheduleXfer(pla['id'], aid, xferBytes,
                                             computeTasksByLayer[prevLayerPtr["LayerId"]][pla['id']]))
                    if plsa >= samplesAssigned + assignment['localBatch']:
                        break
                    plsa += pla["localBatch"]
                
            samplesAssigned += assignment['localBatch']
            computeTasksByLayer[lid][aid] = sim.scheduleCompute(aid, lid, computeTime, prevXferTasks)
        assert(samplesAssigned == totalBatchSize)
    
net = buildHostAndGpuNetwork(2, 2, 10, 10)
net.printAllPaths()
trainingPlan = json.loads(buildSimplePlan())
print(json.dumps(trainingPlan, indent=2, sort_keys=False))
prof_v100 = Profile()
# TODO!!! Insert fake datapoints for testing ..
profiles = {"V100": prof_v100}
simulate(trainingPlan, net, profiles)
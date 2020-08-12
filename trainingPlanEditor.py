import jsonpickle
import json

layers = [None]

class Layer:

    def __init__(self, layerId, modelBytes, assignedAccelerators, prevLayers):
        self.layerId = layerId
        # Layer.nextLayerId += 1
        # self.computeTime = computeTime
        self.modelBytes = modelBytes
        self.assignedAccelerators = assignedAccelerators
        self.prevLayers = prevLayers                    # [(LayerId, inputByteSize), ...]
        self.nextLayers = []                            # [(LayerId, outputByteSize), ...]
        
        for prev in prevLayers:
            prevId = prev["LayerId"]
            inputBytes = prev["InputBytesPerSample"]
            layers[prevId].nextLayers.append({"LayerId": self.layerId, "OutputBytesPerSample": inputBytes})
        layers.append(self)

########################################################
### Helper function                                  ###
########################################################
def buildSimplePlan():
    plan = []
    plan.append(Layer(1, 1000, [{"id": 2, "localBatch": 64}], []))
    plan.append(Layer(2, 10000, [{"id": 3, "localBatch": 32}, {"id": 5, "localBatch": 32}],
                     [{"LayerId": 1, "InputBytesPerSample": 100}]))
    return jsonpickle.encode(plan, unpicklable=False)

# print(json.dumps(json.loads(buildSimplePlan()), indent=2, sort_keys=False))
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

import jsonpickle
import json


class Layer:
    
    # layers = [None] # for tracking all layers.
    
    def __init__(self, layerId, name, modelBytes, prevLayers, assignedAccelerators = None):
        self.layerId = layerId
        self.name = name
        self.modelBytes = modelBytes
        self.prevLayers = prevLayers                    # [(LayerId, inputByteSize), ...]
        self.assignedAccelerators = assignedAccelerators
        # self.nextLayers = []                            # [(LayerId, outputByteSize), ...]
        
        # for prev in prevLayers:
        #     prevId = prev["LayerId"]
        #     inputBytes = prev["InputBytesPerSample"]
        #     Layer.layers[prevId].nextLayers.append({"LayerId": self.layerId, "OutputBytesPerSample": inputBytes})
        #
        # Layer.layers.append(self)
        
    def assignAccelerators(self, assignedAccelerators):
        self.assignedAccelerators = assignedAccelerators

########################################################
### Helper function                                  ###
########################################################
def buildSimplePlan():
    plan = []
    plan.append(Layer(1, "fist", 1000, [],[{"id": 2, "localBatch": 64}]))
    plan.append(Layer(2, "second", 10000, [{"LayerId": 1, "InputBytesPerSample": 100}],
                     [{"id": 3, "localBatch": 32}, {"id": 5, "localBatch": 32}]))
    return jsonpickle.encode(plan, unpicklable=False)

# print(json.dumps(json.loads(buildSimplePlan()), indent=2, sort_keys=False))
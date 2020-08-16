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

import json

class Profile:
    def __init__(self, jsonFilepath = None):
        if jsonFilepath:
            self.datapoint = json.load(open(jsonFilepath))
        else:
            self.datapoint = [{}, {}] # [<dict> layerId] = [(localBatch, computeTime), ...]
        
    def addDatapoint(self, layerIdInt, localBatch, computeTimes, alreadySorted = False):
        layerId = str(layerIdInt)
        if layerId not in self.datapoint[0]:
            for i in range(len(self.datapoint)):
                self.datapoint[i][layerId] = []
        assert(len(self.datapoint) == len(computeTimes))
        for i in range(len(self.datapoint)):
            self.datapoint[i][layerId].append((localBatch, computeTimes[i]))
            if not alreadySorted:
                self.datapoint[i][layerId].sort() # TODO: make it efficient when performance matters.
        
    def getCost(self, phase, layerIdInt, localBatch):
        layerId = str(layerIdInt)
        
        batch_a = 0
        compTime_a = 0
        batch_b = 0
        compTime_b = 0
        
        for dpLocalBatch, dpComputeTime in self.datapoint[phase][layerId]:
            if localBatch <= dpLocalBatch:
                batch_b = dpLocalBatch
                compTime_b = dpComputeTime
                break
            else:
                batch_a = dpLocalBatch
                compTime_a = dpComputeTime
        
        assert(batch_b > 0)
        
        return (localBatch - batch_a + 0.0) * (compTime_b - compTime_a + 0.0) / (batch_b - batch_a + 0.0) + compTime_a

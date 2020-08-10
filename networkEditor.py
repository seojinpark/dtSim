import json
import jsonpickle

MAX_ELEMENTS = 100
# These are populated by constructors. Don't add to them manually.
class Network:
    nextGuid = 0
    elements = []
    accelerators = []
    hosts = []
    switches = []
    links = []
    linkIndexed = [[None for x in range(MAX_ELEMENTS)] for y in range(MAX_ELEMENTS)]

net = Network();
netForStr = {"switches": net.switches, "hosts": net.hosts, "accelerators": net.accelerators, "links": net.links}

# Default configurations
DEFAULT_BW_PCIE_TO_GPU = 1000   # in Gbps
DEFAULT_LAT_PCIE_TO_GPU = 10    # in microseconds
DEFAULT_LAT_NIC_TO_HOST = 100   # in microseconds

class Element:
    guid = -1    
    def __init__(self):
        self.guid = net.nextGuid
        net.nextGuid += 1
        net.elements.append(self)
        
class Accelerator(Element):
    model = ""   # GPU model name.
    def __init__(self, model = "V100"):
        Element.__init__(self)
        self.model = model
        net.accelerators.append(self)

# Goal: model memory bw and CPU effects?
class Host(Element):
    sharedMaxPcieBw = 1000      # in Gbps
    
    def __init__(self, sharedMaxPcieBw):
        Element.__init__(self)
        net.hosts.append(self)
        self.sharedMaxPcieBw = sharedMaxPcieBw
    
class Switch(Element):
    bw = 1000   # in Gbps
    lat = 0     # in microseconds
    def __init__(self, bw = -1, lat = 0):
        Element.__init__(self)
        self.bw = bw
        self.lat = lat
        net.switches.append(self)

class Link(json.JSONEncoder):
    src = -1
    dst = -1
    bw = 0
    lat = 0
    def __init__(self, src, dst, bandwidth, latency):
        self.src = src.guid
        self.dst = dst.guid
        self.bw = bandwidth
        self.lat = latency
        net.links.append(self)
        net.linkIndexed[src.guid][dst.guid] = self

##########################################################################
# Helper functions
##########################################################################
def addAccelerators(switch, numberOfGpus,
                    bwPcieToGpu = DEFAULT_BW_PCIE_TO_GPU,
                    latPcieToGpu = DEFAULT_LAT_PCIE_TO_GPU):
    for i in range(numberOfGpus):
        gpu = Accelerator()
        pcieLink1 = Link(switch, gpu, bwPcieToGpu, latPcieToGpu)
        pcieLink2 = Link(gpu, switch, bwPcieToGpu, latPcieToGpu)

def printNetworkConfig():
    # serializedNetwork = json.dumps(net.links)
    # serializedNetwork = jsonpickle.encode(net.links, unpicklable=False)
    serializedNetwork = jsonpickle.encode(netForStr, unpicklable=False)
    print(serializedNetwork)
    # print(json.dumps(netForStr))
    # print(net.links)

##########################################################################
# Sample network builders
##########################################################################
def buildSimpleNetwork():
    # Network with a single switch and GPUs. No host.
    rootSw = Switch()
    addAccelerators(rootSw, 10)
    printNetworkConfig()
    
buildSimpleNetwork()
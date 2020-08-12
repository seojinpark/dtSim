import json
import jsonpickle
import heapq

MAX_ELEMENTS = 100

# Default configurations
DEFAULT_BW_PCIE_TO_GPU = 1000   # in Gbps
DEFAULT_LAT_PCIE_TO_GPU = 10    # in microseconds
DEFAULT_LAT_NIC_TO_HOST = 100   # in microseconds

class Element:
    def __init__(self, net):
        self.guid = net.nextGuid
        net.nextGuid += 1
        net.elements.append(self)
        
class Accelerator(Element):
    # model = ""   # GPU model name.
    def __init__(self, net, model = "V100"):
        Element.__init__(self, net)
        self.model = model              # GPU model name.
        net.accelerators.append(self)

# Goal: model memory bw and CPU effects?
class Host(Element):
    sharedMaxPcieBw = 1000      # in Gbps
    
    def __init__(self, net, sharedMaxPcieBw = 1000):
        Element.__init__(self, net)
        net.hosts.append(self)
        self.sharedMaxPcieBw = sharedMaxPcieBw
    
class Switch(Element):
    def __init__(self, net, bw = -1, lat = 0):
        Element.__init__(self, net)
        self.bw = bw        # in Gbps
        self.lat = lat      # in microseconds
        net.switches.append(self)

class Link(json.JSONEncoder):
    # lid = -1
    # src = -1
    # dst = -1
    # bw = 0
    # lat = 0
    def __init__(self, net, src, dst, bandwidth, latency):
        self.src = src.guid
        self.dst = dst.guid
        self.bw = bandwidth
        self.lat = latency
        self.lid = net.nextLinkId
        net.nextLinkId += 1
        net.links.append(self)
        net.linkFromSrc[src.guid][dst.guid] = self
        net.pathFromSrc[src.guid][dst.guid] = [dst.guid]
    
    def calcXferTime(self, xferBytes):
        return self.lat + xferBytes / self.bw
    

class Network:
    VERBOSE = False

    def __init__(self):
        # These are populated by constructors. Don't add to them manually.
        self.nextGuid = 0
        self.elements = []
        self.accelerators = []
        self.hosts = []
        self.switches = []
        self.nextLinkId = 0
        self.links = []
        self.linkFromSrc = [dict() for x in range(MAX_ELEMENTS)] # [<list> src][<dict> dst] == LinkObject
        self.arePathsReady = False
    
        # Paths are calculated later.
        self.pathFromSrc = [dict() for x in range(MAX_ELEMENTS)] # [<list> src][<dict> dst] == <list> [1st_hop, 2nd_hop, ..., final_hop]
    
    def printConfigInJSON(self):
        states = {"switches": self.switches, "hosts": self.hosts, "accelerators": self.accelerators, "links": self.links}
        return jsonpickle.encode(states, unpicklable=False)
    
    def calcShortestPath(self):
        if self.VERBOSE:
            print("*** Initial paths ***")
            print(self.pathFromSrc)
        for iterations in range(len(self.elements)):
            updated = 0
            for src in self.elements:
                for neighborGuid in self.linkFromSrc[src.guid]:
                    for nextReachable in self.pathFromSrc[neighborGuid]:
                        if nextReachable not in self.pathFromSrc[src.guid]:
                            self.pathFromSrc[src.guid][nextReachable] = [neighborGuid] + self.pathFromSrc[neighborGuid][nextReachable]
                            updated += 1
            if self.VERBOSE:
                print("Completed iter %d. Updated: %d" % (iterations, updated))
                print(self.pathFromSrc)
            if updated == 0:
                break
        self.arePathsReady = True
                
    def printAllPaths(self):
        for src in range(len(self.elements)):
            print("From %3d (%s) ===> to" % (src, type(self.elements[src]).__name__))
            for dst in self.pathFromSrc[src]:
                print("             %3d (%s) :  %s" %
                     (dst, type(self.elements[dst]).__name__, str(self.pathFromSrc[src][dst])))


##########################################################################
# Network Simulation
##########################################################################
class Task:
    def __init__(self, incompletePrevTaskCount):
        # Note: all times are in microseconds.
        self.readyTime = -1       # time when this task is ready to run.
                             # It will be updated everytime each previous task is completed & track the latestest value.
                             # '-1' means unset yet.
        self.startTime = None
        self.finishTime = None
        self.nextTasks = []
        self.incompletePrevTaskCount = incompletePrevTaskCount
        
        # self.readyTime = readyTime
        
    def registerNextTask(self, taskDependingOnThis):
        self.nextTasks.append(taskDependingOnThis)
    
    def __lt__(self, other):
        return self.readyTime < other.readyTime

class ComputeTask(Task):
    # acceleratorGuid = -1
    # layerId = -1
    # computeTime = 0
    def __init__(self, prevTaskCount, acceleratorGuid, layerId, computeTime):
        Task.__init__(self, prevTaskCount)
        self.acceleratorGuid = acceleratorGuid
        self.layerId = layerId
        self.computeTime = computeTime

class NetworkTask(Task):
    # linkId = -1
    # xferBytes = 0
    def __init__(self, prevTaskCount, linkId, xferBytes):
        Task.__init__(self, prevTaskCount)
        self.linkId = linkId
        self.xferBytes = xferBytes
    
# Currently, it doesn't support bw limit from host or switch. Latency is considered.
class Simulation:
    VERBOSE = True

    def __init__(self, network):
        assert(network.arePathsReady)
        self.net = network
        self.linkTasks = [] # Probably not needed in Python ...
        self.compTasks = [] # Probably not needed in Python ...
        self.initialTasks = []
        # self.linkReadyTime = [0] * len(network.links)
        # self.accelReadyTime = [0] * len(network.elements)

    # Returns the final link transfer task.
    def scheduleXfer(self, src, dst, xferBytes, prevComputeTask = None):
        path = self.net.pathFromSrc[src][dst]
        prevNode = src
        prevTask = prevComputeTask
        for nextNode in path:
            link = self.net.linkFromSrc[prevNode][nextNode]
            task = NetworkTask(0 if prevTask == None else 1, link.lid, xferBytes)
            if prevTask == None:
                self.initialTasks.append(task)
                task.readyTime = 0
            self.linkTasks.append(task)
            prevTask.registerNextTask(task)
            prevNode = nextNode
            prevTask = task
        return prevTask

    # Returns compute task.
    def scheduleCompute(self, acceleratorId, layerId, computeTime, prevXferTasks = []):
        task = ComputeTask(len(prevXferTasks), acceleratorId, layerId, computeTime)
        if len(prevXferTasks) == 0:
            self.initialTasks.append(task)
            task.readyTime = 0
        self.compTasks.append(task)
        for linkTask in prevXferTasks:
            linkTask.registerNextTask(task)
        return task

    def run(self):
        taskq = [(t.readyTime, t) for t in self.initialTasks]
        linkReadyTime = [0] * len(self.net.links)  # [linkId] = Microseconds when link becomes free.
        accelReadyTime = [0] * len(self.net.elements) # [guid] = Microseconds when accelerator becomes free.
        
        heapq.heapify(taskq)
        if self.VERBOSE:
            print("Initial task: " + jsonpickle.encode(taskq, unpicklable=False))

        while len(taskq) > 0:
            readyTime, task = heapq.heappop(taskq)
            assert(task.startTime == None)
            assert(task.finishTime == None)
            assert(readyTime == task.readyTime)
            
            
            if isinstance(task, ComputeTask):
                task.startTime = max(readyTime, accelReadyTime[task.acceleratorGuid])
                task.finishTime = task.startTime + task.computeTime
                accelReadyTime[task.acceleratorGuid] = task.finishTime
                
                for nextTask in task.nextTasks:
                    nextTask.readyTime = max(nextTask.readyTime, task.finishTime)
                    nextTask.incompletePrevTaskCount -= 1                    
                    assert(nextTask.incompletePrevTaskCount >= 0)
                    
                    if nextTask.incompletePrevTaskCount == 0:
                        heapq.heappush(taskq, (nextTask.readyTime, nextTask))
                        
            elif isinstance(task, NetworkTask):
                link = self.net.links[task.linkId]
                assert(link.lid == task.linkId)
                task.startTime = max(readyTime, linkReadyTime[task.linkId])
                assert(task.xferBytes > 0)
                task.finishTime = task.startTime + link.calcXferTime(task.xferBytes)
                linkReadyTime[task.linkId] = task.finishTime - link.lat # A link can take new ingress data before done with egress work.
                
                for nextTask in task.nextTasks:
                    if isinstance(nextTask, NetworkTask):
                        nextTask.readyTime = max(nextTask.readyTime, task.startTime + link.lat)
                    elif isinstance(nextTask, ComputeTask):
                        nextTask.readyTime = max(nextTask.readyTime, task.finishTime)
                    else:
                        assert(False)
                    
                    nextTask.incompletePrevTaskCount -= 1
                    assert(nextTask.incompletePrevTaskCount >= 0)
                    if nextTask.incompletePrevTaskCount == 0:
                        heapq.heappush(taskq, (nextTask.readyTime, nextTask))
            # self.dumpInternalState()
        if self.VERBOSE:
            print("simulation completed.")
            self.dumpInternalState()
            print("")

    def dumpInternalState(self):
        # print("Dumping internal states...")
        print("# readyTime  startTime  finalTime   taskType                                         nextTasks")
        for t in self.compTasks + self.linkTasks:
            print("%10d %10s %10s    %s    %s"
                % (t.readyTime, str(t.startTime), str(t.finishTime), str(t), str(t.nextTasks)))
        
            
        
##########################################################################
# Helper functions
##########################################################################
def addAccelerators(net, switchOrHost, numberOfGpus,
                    bwPcieToGpu = DEFAULT_BW_PCIE_TO_GPU,
                    latPcieToGpu = DEFAULT_LAT_PCIE_TO_GPU):
    for i in range(numberOfGpus):
        gpu = Accelerator(net)
        pcieLink1 = Link(net, switchOrHost, gpu, bwPcieToGpu, latPcieToGpu)
        pcieLink2 = Link(net, gpu, switchOrHost, bwPcieToGpu, latPcieToGpu)

def sanityCheck(net):
    print(net.printConfigInJSON())
    net.calcShortestPath()
    net.printAllPaths()

##########################################################################
# Sample network builders
##########################################################################
def buildSimpleNetwork():
    # Network with a single switch and GPUs. No host.
    simpleNet = Network()
    rootSw = Switch(simpleNet)
    addAccelerators(simpleNet, rootSw, 2)
    # print(simpleNet.printConfigInJSON())
    return simpleNet

def buildHostAndGpuNetwork(hostCount, gpusPerHost, hostToTorBw, hostToTorLat):
    # Network with a single switch and GPUs. No host.
    net = Network()
    rootSw = Switch(net)
    for i in range(hostCount):
        host = Host(net)
        nicLink1 = Link(net, rootSw, host, hostToTorBw, hostToTorLat)
        nicLink2 = Link(net, host, rootSw, hostToTorBw, hostToTorLat)
        addAccelerators(net, host, gpusPerHost)
    net.calcShortestPath()
    return net

##########################################################################
# Tests
##########################################################################
def __testSimulationBasic():
    net = buildSimpleNetwork()
    net.calcShortestPath()
    # net.printAllPaths()
    
    sim = Simulation(net)
    tasks = []
    tasks.append(sim.scheduleCompute(1, 1, 100, []))
    # print("Scheduled: %s" % str(tasks[-1]))
    # print(jsonpickle.encode(tasks[-1], unpicklable=False) + str(tasks[-1].nextTasks))
    tasks.append(sim.scheduleXfer(1, 2, 1000, tasks[-1]))
    # print("Scheduled: %s" % str(tasks[-1]))
    # print(jsonpickle.encode(tasks[-1], unpicklable=False) + str(tasks[-1].nextTasks))
    tasks.append(sim.scheduleCompute(2, 2, 100, [tasks[-1]]))
    # print("Scheduled: %s" % str(tasks[-1]))
    # print(jsonpickle.encode(tasks[-1], unpicklable=False) + str(tasks[-1].nextTasks))
    sim.run()


# net = buildHostAndGpuNetwork(2, 2, 10, 10)
# sanityCheck(net)
__testSimulationBasic()
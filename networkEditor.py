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
import jsonpickle
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
# from grave import plot_network
# from grave.style import use_attributes

MAX_ELEMENTS = 100

# Default configurations
DEFAULT_BW_NVLINK = 1600   # in Gbps
DEFAULT_BW_PCIE_TO_GPU = 1000   # in Gbps
DEFAULT_LAT_NVLINK = 10   # in microseconds
DEFAULT_LAT_PCIE_TO_GPU = 17    # in microseconds
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
        self.rank = len(net.accelerators)
        net.accelerators.append(self)

    def __str__(self):
        return str(self.guid) + "A"


# Goal: model memory bw and CPU effects?
class Host(Element):
    sharedMaxPcieBw = 1000      # in Gbps
    
    def __init__(self, net, sharedMaxPcieBw = 1000):
        Element.__init__(self, net)
        net.hosts.append(self)
        self.sharedMaxPcieBw = sharedMaxPcieBw
    
    def __str__(self):
        return str(self.guid) + "H"
    
class Switch(Element):
    def __init__(self, net, bw = -1, lat = 0):
        Element.__init__(self, net)
        self.bw = bw        # in Gbps
        self.lat = lat      # in microseconds
        net.switches.append(self)
    
    def __str__(self):
        return str(self.guid) + "S"

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
                     
    def plotNetwork(self, showPlot=True):
        g = nx.DiGraph()
        g.add_nodes_from(self.elements)
        for link in self.links:
            g.add_edge(self.elements[link.src], self.elements[link.dst], object=link)
        # nx.draw(g)
        nodeColors = []
        nodeShapes = []
        for n in g:
            if isinstance(n, Accelerator):
                nodeColors.append('gray')
                nodeShapes.append('s')
            elif isinstance(n, Host):
                nodeColors.append('orange')
                nodeShapes.append('o')
            elif isinstance(n, Switch):
                nodeColors.append('red')
                nodeShapes.append('o')
            else:
                nodeColors.append('yellow')
        if showPlot:
            nx.draw(g, with_labels=True, node_color=nodeColors, node_shape='s', font_weight='bold')
            plt.show()
        return g


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
        self.log_tasksByGuid = [list() for x in range(len(network.elements))]
        # self.linkReadyTime = [0] * len(network.links)
        # self.accelReadyTime = [0] * len(network.elements)
    
    def plotOnClick(self, event):
        print("Node on plot was clicked.")
        if isinstance(event.artist, PathCollection):
            all_nodes = event.artist
            ind = event.ind[0] # event.ind is a single element array.
            print("Accelerator %d ran following tasks: " % self.display_accelerators[ind].guid)
            guid = self.display_accelerators[ind].guid
            print("#     Task  readyTime  startTime  finalTime       nextTasks")
            for t in self.log_tasksByGuid[guid]:
                # print(jsonpickle.encode(task, unpicklable=False))
                if isinstance(t, ComputeTask):
                    print("%10s %10.1f %10.1f %10.1f     %s"
                        % ("layer" + str(t.layerId), t.readyTime, t.startTime, t.finishTime, str(t.nextTasks)))
                elif isinstance(t, NetworkTask):
                    link = self.net.links[t.linkId]
                    print("%10s %10.1f %10.1f %10.1f     %s"
                        % ("%d->%d"%(link.src, link.dst), t.readyTime, t.startTime, t.finishTime, str(t.nextTasks)))
    
    def plotNetwork(self):
        g = self.net.plotNetwork(showPlot=False)
        pos = nx.layout.spring_layout(g)
        
        nodeColors = []
        nodeShapes = []
        accelerators = []
        hosts = []
        switches = []
        for n in g:
            if isinstance(n, Accelerator):
                accelerators.append(n)
            elif isinstance(n, Host):
                hosts.append(n)
            elif isinstance(n, Switch):
                switches.append(n)
            else:
                assert(False)
        
        fig, ax = plt.subplots()
        node_size = 300
        anodes = nx.draw_networkx_nodes(g, pos, nodelist=accelerators, node_shape='s', node_color='white', edgecolors='black')
        anodes.set_picker(5)
        hnodes = nx.draw_networkx_nodes(g, pos, nodelist=hosts, node_shape='o', node_color='orange', edgecolors='black')
        snodes = nx.draw_networkx_nodes(g, pos, nodelist=switches, node_shape='o', node_color=(1,153./255,153./153), edgecolors='black')

        nx.draw_networkx_edges(g, pos, node_size=node_size, arrowstyle='->',
                                        arrowsize=15, edge_color='black', width=1)
        nx.draw_networkx_labels(g, pos, font_color='black', font_family='arial',
                                        font_size=10)
        
        # nx.draw_networkx_nodes(g, with_labels=True, node_color=nodeColors, node_shape='s', font_weight='bold')
        # art.set_picker(10)
        self.display_accelerators = accelerators
        fig.canvas.mpl_connect('pick_event', self.plotOnClick)
        
        plt.show()
        return g

    # Returns the final link transfer task.
    def scheduleXfer(self, src, dst, xferBytes, prevComputeTask = None):
        if src == dst:
            return prevComputeTask

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
                # For logging purpose, register the current task to the used element.
                self.log_tasksByGuid[task.acceleratorGuid].append(task)
                
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
                
                # For logging purpose, register the current task to the used element.
                self.log_tasksByGuid[link.src].append(task)
                self.log_tasksByGuid[link.dst].append(task)
                
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
            print("%10.1f %10.1f %10.1f    %s    %s"
                % (t.readyTime, t.startTime, t.finishTime, str(t), str(t.nextTasks)))



##########################################################################
# Helper functions
##########################################################################
def addAccelerators(net, switchOrHost, numberOfGpus,
                    nvlinkBwAmongGpus = None,
                    bwPcieToGpu = DEFAULT_BW_PCIE_TO_GPU,
                    latPcieToGpu = DEFAULT_LAT_PCIE_TO_GPU):
    gpus = []
    for i in range(numberOfGpus):
        gpu = Accelerator(net)
        gpus.append(gpu)
        pcieLink1 = Link(net, switchOrHost, gpu, bwPcieToGpu, latPcieToGpu)
        pcieLink2 = Link(net, gpu, switchOrHost, bwPcieToGpu, latPcieToGpu)
    
    if nvlinkBwAmongGpus != None:
        for gpu1 in gpus:
            for gpu2 in gpus:
                if gpu1 == gpu2:
                    continue
                Link(net, gpu1, gpu2, nvlinkBwAmongGpus, DEFAULT_LAT_NVLINK)
        

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

def buildAwsP3Network(hostCount, gpusPerHost, hostToTorBw, hostToTorLat):
    # Network with a single switch and GPUs. No host.
    net = Network()
    rootSw = Switch(net)
    for i in range(hostCount):
        host = Host(net)
        nicLink1 = Link(net, rootSw, host, hostToTorBw, hostToTorLat)
        nicLink2 = Link(net, host, rootSw, hostToTorBw, hostToTorLat)
        addAccelerators(net, host, gpusPerHost, 1600)
    net.calcShortestPath()
    return net

##########################################################################
# Tests
##########################################################################
def __testSimulationBasic():
    net = buildSimpleNetwork()
    net.calcShortestPath()
    # net.plotNetwork()
    # net.printAllPaths()
    
    sim = Simulation(net)
    tasks = []
    sim.scheduleXfer(1, 1, 100, prevComputeTask = None)
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

def main():
    # net = buildHostAndGpuNetwork(2, 2, 10, 10)
    # sanityCheck(net)
    # net.plotNetwork()

    # net = buildAwsP3Network(2, 4, 10, 10)
    # sanityCheck(net)
    # net.plotNetwork()

    __testSimulationBasic()

if __name__ == "__main__":
    main()

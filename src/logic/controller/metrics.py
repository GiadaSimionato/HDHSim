from shapely.geometry import Point
from shapely.ops import unary_union
import numpy as np

from src.simulator import Hook
from src.logger import GlobalLogger

class CoverageMetric(Hook):
    def __init__(self, cfg):
        self.startArea = 0.0
        self.unionNodes = 0
        self.unionNodesNotFaulty = 0
        self.currentC = 0.0
        self.endTick = 0
        self.rn = cfg.params.rs[0]
        self.ra = cfg.params.rs[1]
        self.th = cfg.params.th_metric

    def start(self, objects):
        nodes = objects.loc[(objects["type"]=="node") & (objects["fault"]==False)]
        nodes = np.asarray(nodes[["x", "y"]])
        circles = []
        for elem in nodes:
            circles.append(Point(elem[0], elem[1]).buffer(self.rn))
        self.unionNodesNotFaulty = unary_union(circles)

        nodes = objects.loc[(objects["type"]=="node") & (objects["fault"]==True)]
        nodes = np.asarray(nodes[["x", "y"]])
        circles = []
        for elem in nodes:
            circles.append(Point(elem[0], elem[1]).buffer(self.rn))
        self.unionNodes = unary_union([self.unionNodesNotFaulty, unary_union(circles)])
        self.startArea = (self.unionNodes.difference(self.unionNodesNotFaulty)).area
    
    def tick(self, objects):
        agents = objects.loc[(objects["type"]=="agent") & (objects["state"]==2)]
        agents = np.asarray(agents[["x", "y"]])
        circles_agents = []
        for elem in agents:
            circles_agents.append(Point(elem[0], elem[1]).buffer(self.ra))
        union_agents = unary_union(circles_agents)

        area_actual = (self.unionNodes.difference(unary_union([self.unionNodesNotFaulty, union_agents]))).area
        self.currentC = 1-(area_actual/self.startArea)

        GlobalLogger.log("coverage", self.currentC)
        GlobalLogger.log("bound_agents", len(agents))

        if(self.currentC>=self.th):
            #print("Stopping condition met")
            pass


    def end(self, object):
        pass
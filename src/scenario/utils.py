from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import numpy as np
import tempfile
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class Node:

    def __init__(self, id, x, y, heading):
        self.id = id                # node id
        self.x = x                  # x coordinate
        self.y = y                  # y coordinate
        self.heading = heading      # heading [rad]
        self.fault = False          # True if fault, False otherwise
        self.boundary = False       # True if boundary, False otherwise
        self.ng_to_fault = False    # True if neighbour to faulty node, False otherwise
        self.ng_to_boundary = False # True if neighbour to frontier node, False otherwise
        self.neighbours = []        # List of neighbours (Node objects)

def normalizeIntervals(intervals):
    norm_ints = []
    for i in range(len(intervals)):
        if intervals[i,0]<0:
            norm_ints.append([0, intervals[i,1]])
            norm_ints.append([2*np.pi + intervals[i,0], 2*np.pi])
        elif intervals[i,1]>2*np.pi:
            norm_ints.append([intervals[i,0], 2*np.pi])
            norm_ints.append([0, intervals[i,1]-2*np.pi])
        else:
            norm_ints.append(list(intervals[i]))

    return norm_ints

def mergeIntervals(intervals):
    intervals.sort()
    stack = []
    stack.append(intervals[0])
    for i in intervals[1:]:
        if stack[-1][0] <= i[0] <= stack[-1][-1]:   # if overlap
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)
    return stack

# Create list of node objects and set boundary flag
def setBoundary(points, rs):
    k=0.0
    coords = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        coords[i,0] = point.x
        coords[i,1] = point.y
    dist = distance_matrix(coords, coords)
    max_dist = np.full((len(points), len(points)), 2*rs+k)
    adj_mat = dist<max_dist
    np.fill_diagonal(adj_mat, False)

    nodes = []
    for i in range(len(points)):    #for each point
        heading = np.pi*np.random.uniform(0,2)
        node = Node(len(nodes), coords[i,0], coords[i,1], heading)
        indices = np.squeeze(np.argwhere(adj_mat[i]), axis=-1)
        ng = coords[indices]
        pos = np.tile(coords[i].reshape(-1,1).T, (len(ng), 1))
        diff = np.subtract(ng,pos)
        rab = np.zeros(diff.shape)
        rab[:,0] = np.linalg.norm(diff, axis=1)
        rab[:,1] = np.arctan2(diff[:,1], diff[:,0])%(2*np.pi)

        phi = np.divide(np.power(rab[:,0],2), 2*rs*rab[:,0])
        phi = np.arccos(phi)    # in [0,pi]
        intervals = np.zeros((len(phi), 2))
        intervals[:,0] = np.subtract(rab[:,1], phi)
        intervals[:,1] = np.add(rab[:,1], phi)
        intervals = normalizeIntervals(intervals)
        intervals = np.asarray(mergeIntervals(intervals))
        c = np.sum(np.subtract(intervals[:,1], intervals[:,0]))/(2*np.pi)
        
        if(c<0.98):
            node.boundary = True
        else:
            node.boundary = False
        nodes.append(node)

    return nodes
    
        
        

def areAdjacent(x1, y1, x2, y2, dist):
    eps=0.05
    rd = np.sqrt(np.power(x1-x2,2) + np.power(y1-y2,2))
    #return np.abs(rd-dist)<eps
    return rd <= dist-eps

# Set neighbours
def setNeighbours(nodes, r_sensing):

    for node_i in nodes:
        for node_j in nodes:
            if node_i.id!=node_j.id and areAdjacent(node_i.x, node_i.y, node_j.x, node_j.y, 2*r_sensing):
                node_i.neighbours.append(node_j)
    return nodes

# --- Removes from the neighbours the nodes no longer part of the network ---
# @param nodes: list of Node objects
# @return nodes: list of Node objects with neighbours updated
def updateNeighbours(nodes):
    nodes_id = [node.id for node in nodes]
    for node in nodes:
        new_ng = []
        for ng in node.neighbours:
            if ng.id in nodes_id:
                new_ng.append(ng)
        node.neighbours = new_ng
    return nodes

# --- Checks whether a path contains one or more duplicates of some nodes.
# @param path: list of Node objects
# @return bool: True if the list doesn't contain duplicates, False otherwise
def isValid(path):
    ids = [x.id for x in path]
    ids_set = set(ids)
    return len(ids)==len(ids_set)

# --- Find the group of nodes to break.
# @param paths: list of lists of Node objects, paths of length L
# @return list of Node objects to break
def findGroup(paths):
    valid_paths = []
    for path in paths:
        if isValid(path):
            valid_paths.append(path)
    index = random.randint(0, len(valid_paths)-1)
    return valid_paths[index]

# --- Finds all the path of length L starting from node.
# @param node: Node object, starting node of the path
# @param L: int, length of path
# @return paths: list of lists of Node objects, paths of length L starting from node
def findPaths(node, L):
    if L==0:
        return [[node]]
    paths = [[node] + path for ng in node.neighbours for path in findPaths(ng, L-1)]
    return paths

# --- Returns the index of the node in nodes that has id as id.
# @param nodes: list of Node objects
# @param id: int
# @return int: index of element in nodes that has as id, id
def getId(nodes, id):
    h = [x.id for x in nodes]
    return h.index(id)

# --- Utils function to find paths. Connected to getConnectedComponents.
def DFS(temp, v, visited, nodes):
    visited[v] = True
    temp.append(v)
    for i in nodes[v].neighbours:
        a = getId(nodes, i.id)
        if visited[a] == False:
            temp = DFS(temp, a, visited, nodes)
    return temp

# --- Returns all the connected commponents of a graph whose length >=L
# @param nodes: list of Node objects
# @param L: minimum length of connected component
# @return tcc: list of lists of Node objects (list of connected components) with length >=L
def getConnectedComponents(nodes, L):
    visited = []
    cc = []
    for i in range(len(nodes)):
        visited.append(False)
    for v in range(len(nodes)):
        if visited[v] == False:
            temp = []
            cc.append(DFS(temp, v, visited, nodes))
    tcc = []
    for c in cc:
        tc = []
        for elem in c:
            tc.append(nodes[elem])
        if len(tc)>=L:  # return only connected components with length >= L
            random.shuffle(tc)  # shuffle for randomization
            tcc.append(tc)
    return tcc


            
# Render network
def renderNetwork(nodes, r_sensing):
    margin = 30
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.plot(3,5)
    x_values = [node.x for node in nodes]
    y_values = [node.y for node in nodes]
    ax.set_xlim([min(x_values)-margin, max(x_values)+margin])
    ax.set_ylim([min(y_values)-margin, max(y_values)+margin])
    for node in nodes:
        #plt.text(node.x, node.y, node.id)
        if node.fault:
            plt.plot(node.x, node.y, 'r.')
            ax.add_artist(plt.Circle((node.x,node.y), r_sensing, fill=False, color='r', ls='--'))
        elif node.boundary:
            plt.plot(node.x, node.y, 'g.')
            ax.add_artist(plt.Circle((node.x,node.y), r_sensing, fill=False, color='g', ls='--'))
        else:
            plt.plot(node.x, node.y, 'k.')
            ax.add_artist(plt.Circle((node.x,node.y), r_sensing, fill=False, color='k', ls='--'))
        for ng in node.neighbours:
            plt.plot([node.x, ng.x], [node.y, ng.y], color='c', ls='-', linewidth='0.3') #draw line joining two points
    
    plt.gca().set_aspect('equal', adjustable='box')
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(f.name, dpi=300)
    return f.name


# Sample a point within area
def samplePoint(area):
    minx, miny, maxx, maxy = area.bounds
    nIter=1000
    i = 0
    while i<nIter:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if area.contains(pnt):
            return pnt
        i += 1
    raise Exception("Cannot produce a sample in area within {} iterations".format(nIter))

# Check if point pnt is far enough (d) from others in points
def validPoint(pnt, points, d, eps):
    for p in points:
        if np.sqrt((pnt.x-p.x)**2+(pnt.y-p.y)**2)<=d+eps:
            return False
    return True

# Get percentage coverage of area
def getCoverage(area, points, rs):

    circles = []
    for point in points:
        circles.append(point.buffer(rs))
    union = unary_union(circles)
    return union
# Find first element of list1 that is present in list2 but not in list3
def common(list1, list2, list3):
    for elem in list1:
        if elem in list2 and not (elem in list3):
            return elem
    return -1

def findHole(cc, ntb, rs):
    ids = []
    index = random.randint(0, len(cc)-1) #random selection of starting node
    ids.append(cc[index].id)
    if ntb==1:
        return ids
    index = random.randint(0, len(cc[index].neighbours)-1)
    ids.append(cc[getId(cc, ids[0])].neighbours[index].id)
    if ntb==2:
        return ids
    ntb -= 2
    for i in range(ntb):
        found = False
        for n1 in ids:
            if not found:
                for n2 in ids:
                    if n1!=n2 and areAdjacent(cc[getId(cc, n1)].x, cc[getId(cc, n1)].y, cc[getId(cc, n2)].x, cc[getId(cc, n2)].y, 2*rs):
                        # Check for common ng
                        ng1 = cc[getId(cc, n1)].neighbours
                        ng1 = [n.id for n in ng1]
                        ng2 = cc[getId(cc, n2)].neighbours
                        ng2 = [n.id for n in ng2]
                        result = common(ng1, ng2, ids)
                        if result != -1:
                            ids.append(result)
                            found = True
                            break
            else:
                break
        if not found:
            raise Exception("Impossible to create hole.")
    return ids

# --- Saves scenario in csv to txt file in path. ---
# id, coordx, coordy, heading, fault, boundary
def saveScenario(nodes, path_save):
    f = open(path_save, "w")
    for node in nodes:
        s = str(node.id)+','+str(node.x)+','+str(node.y)+','+str(node.heading)+','+str(int(node.fault))+','+str(int(node.boundary))+'\n'
        f.write(s)
    f.close()



def scaleData(nodes, margin, size):
    data = []
    for node in nodes:
        data.append([
            node.id,
            node.x,
            node.y,
            node.heading,
            node.fault,
            node.boundary
    ])
    data = np.asarray(data)
    min_x = np.amin(data[:,1])
    min_y = np.amin(data[:,2])
    max_x = np.amax(data[:,1])
    max_y = np.amax(data[:,2])
    
    margin = np.asarray(margin)
    size = np.asarray(size)
    size = size-margin

    origin = [size[0]/2, size[1]/2]
    x_lim = [0, size[0]]
    y_lim = [0, size[1]]

    # Center data on origin
    # data[:,1] -= (max_x+min_x)/2
    # data[:,2] -= (max_y+min_y)/2

    # Scaling data
    scale_d = np.sqrt(pow(data[0][1]-data[1][1],2)+pow(data[0][2]-data[1][2],2))
    
    data[:,1] = (((data[:,1] - min_x)/(max_x - min_x))*x_lim[1])+(margin[0]/2)
    data[:,2] = ((data[:,2] - min_y)/(max_y - min_y))*y_lim[1]+(margin[1]/2)
    scale_d /= np.sqrt(pow(data[0][1]-data[1][1],2)+pow(data[0][2]-data[1][2],2))

    for node, row in zip(nodes, data):
        node.id = row[0]
        node.x = row[1]
        node.y = row[2]
        node.heading = row[3]
        node.fault = row[4]
        node.boundary = row[5]

    return scale_d, nodes
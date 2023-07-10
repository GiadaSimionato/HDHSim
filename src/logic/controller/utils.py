import numpy as np
from src.simulator import GlobalConfig
import random
from scipy.spatial import distance_matrix

def heardHole(readings):

    if(not np.any(readings)):   # if no readings collected
        return False
    
    readings = readings[np.logical_not(np.logical_and(readings[:,1]==1, readings[:,2]!=2))] # filter readings for only nodes or agents bound
    return 1 in readings[:,4]

def filterForAdjacency(readings, rs, mytype):
    k = 0.0
    d_max = rs[:,tuple(map(int,readings[:,1]))][0] + rs[0][mytype] + k

    return readings[readings[:,5]<=d_max]

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

# return a value in [0.0, 1.0] representing the angular coverage ratio (ACR)
def getAngularCoverage(readings, rs, mytype, frontier):

    if(frontier):               # if the node is on the frontier of the area (agents has set this parameter false by default)
        return 1.0
    if(not np.any(readings)):   # if no readings collected
        return 0.0
    
    readings = readings[np.logical_not(np.logical_and(readings[:,1]==1, readings[:,2]!=2))] # filter readings for only nodes or agents bound
    rs_new = np.tile(rs, (len(readings),1))
    readings = filterForAdjacency(readings, rs_new, mytype) # filter readings for adjacent only

    rs_new = np.tile(rs, (len(readings),1))
    phi = np.divide(rs_new[0][mytype]**2 + np.power(readings[:,5],2) -np.power(rs_new[:,tuple(map(int,readings[:,1]))][0], 2), 2*rs_new[0][mytype]*readings[:,5])
    phi = np.arccos(phi)    # in [0,pi]
    intervals = np.zeros((len(readings), 2))
    intervals[:,0] = np.subtract(readings[:,6], phi)
    intervals[:,1] = np.add(readings[:,6], phi)
    intervals = normalizeIntervals(intervals)
    intervals = np.asarray(mergeIntervals(intervals))
    c = np.sum(np.subtract(intervals[:,1], intervals[:,0]))/(2*np.pi)
    
    return c

# Perturbate R&B sensor
def perturbate(readings, params):

    perc_D = params.errors.rab_dist_more
    st_dev_angle = params.errors.rab_angle / 3

    std_D = readings[:, 5] * perc_D / 3
    d_noises = np.random.normal(0.0, std_D, len(readings))
    readings[:, 5] = readings[:, 5] + d_noises

    d_noises = np.random.normal(0.0, st_dev_angle, len(readings))
    readings[:, 6] = np.add(readings[:, 6], d_noises) % (2 * np.pi)
    return readings
    


# N.B. Called only by nodes or bound agents. No need to check if entity is entitled to compute its level.
def updateLevel(node):
    
    readings = node.range_and_bearing
    params = GlobalConfig.cfg().params

    readings = perturbate(readings, params)

    rs = list(params.rs)
    type_ = int(node.type == "agent")

    c = getAngularCoverage(readings, rs, type_, node.boundary)
    if c<params.th_c:  # not covered enough, boundary of holes --> level 0
        return 0
    readings = readings[np.logical_not(np.logical_and(readings[:,1]==1, readings[:,2]!=2))] # filter readings for only nodes or agents bound
    rs_new = np.tile(rs, (len(readings),1))
    readings = filterForAdjacency(readings, rs_new, type_) # filter readings for adjacent only
    readings = readings[readings[:,3]!=-1]  # to exclude those still without level
    if(not np.any(readings)):
        return node.level
    return np.min(readings[:,3])+1
    
def updateGossip(node):
    readings = node.range_and_bearing
    if np.any(readings[:,4]==1.0):
        node.gossip=1
    else: node.gossip=0
    
    

# Driving control: from distance d, angle phi to target, returns v, w
def target2control(d, phi, node):
    params = GlobalConfig.cfg().params
    if(phi > np.pi):
        phi = 2*np.pi - phi
        w_bb = -params.steering_speed_cruise
    else:
        w_bb = params.steering_speed_cruise
    if(phi<=params.th_turn):
        #pure translation
        delta_d = params.driving_speed_cruise*params.dt
        if(d<delta_d):
            return d/params.dt, 0    # this can cause slowdowns near the target point during follow min
        else:
            return params.driving_speed_cruise, 0
        pass
    else:
        delta_phi = params.steering_speed_cruise*params.dt
        if(delta_phi > phi+params.th_turn):
            return 0, phi/params.dt
        else:
            return 0, w_bb


# N.B. f_readings contains only readings of nodes or bound agents
# return distance and angle to target
def followMin(f_readings):
    min_level = np.min(f_readings[:,3])
    f_readings = f_readings[f_readings[:,3]==min_level]             # policy: nearest point with minimum level
    d_min, phi_min = f_readings[np.argmin(f_readings[:,5]), 5:7]    # get direction and distance to cover
    return d_min, phi_min




# return list of lists [index_parents_1, index_parents2] (indices wrt readings)
def getPairParents(readings, rs, additional):

    k = 0.0
    dist = distance_matrix(readings[:,7:9], readings[:,7:9])
    r = rs[:,tuple(map(int,readings[:,1]))][0]
    max_dist = r[:,np.newaxis] + r[np.newaxis,:]+k
    if(additional):
        adj_mat = np.logical_and(dist>=max_dist, dist<=(max_dist+2*rs[0][1]))
    else:
        adj_mat = dist<max_dist
    np.fill_diagonal(adj_mat, False)
    return np.argwhere(np.triu(adj_mat))

def getSolutions(readings, parents, params):

    if(not np.any(readings) or not np.any(parents)):   # if no readings collected or no parents found
        return parents, parents, parents
    
    t_node = []
    t_agent = []
    t_mixed = []

    rs = list(params.rs)
    rot_mat = np.array([[0, -1], [1, 0]])

    structure = np.zeros((len(parents), 6))
    structure[:,0] = readings[parents[:,0],1]
    structure[:,1] = readings[parents[:,1],1]
    structure[:,2] = readings[parents[:,0],7]
    structure[:,3] = readings[parents[:,0],8]
    structure[:,4] = readings[parents[:,1],7]
    structure[:,5] = readings[parents[:,1],8]

    same_p = structure[structure[:,0]==structure[:,1],:]
    diff_p = structure[structure[:,0]!=structure[:,1],:]
    

    if(len(same_p)):    # computations for same type parents
        k = same_p[:,2:4]-same_p[:,4:6]
        d = np.linalg.norm(k, axis=1)
        k /= np.reshape(d, (-1,1))
        db = d/2
        kp = np.matmul(k, rot_mat)
        r = np.tile(rs, (len(same_p),1))
        r = r[:,tuple(map(int,same_p[:,1]))][0]
        dh = np.sqrt(np.power(r,2) - 0.25*np.power(d,2)) + rs[1]
        
        t1_same = same_p[:,4:6] + k*np.reshape(db, (-1,1)) + kp*np.reshape(dh, (-1,1))
        t2_same = same_p[:,4:6] + k*np.reshape(db, (-1,1)) - kp*np.reshape(dh, (-1,1))
    
        t_node = np.vstack((t1_same[same_p[:,0]==0,:], t2_same[same_p[:,0]==0,:]))
        t_agent = np.vstack((t1_same[same_p[:,0]==1,:], t2_same[same_p[:,0]==1,:]))
 

    if(len(diff_p)):    # computations for different type parents
        mask = diff_p[:,0] == 1 # mask for swapping node/agent: if True then swap
        if(np.any(mask)):   # if mask composed by only False, then diff_p[mask] = [] --> error
            diff_p[mask, :] = diff_p[mask, [1,0,4,5,2,3]]   # the agent is last
        k = diff_p[:,2:4]-diff_p[:,4:6]
        d1 = np.linalg.norm(k, axis=1)
        k /= np.reshape(d1, (-1,1))
        kp = np.matmul(k, rot_mat)
        d2 = 2*np.sqrt(rs[1]**2 - np.power((np.power(d1,2) - rs[0]**2 - rs[1]**2)/(2*rs[0]),2))
        p = (np.add(2*d1, d2))/2
        A = np.sqrt(np.multiply(np.multiply(p, np.power(np.subtract(p, d1),2)), np.subtract(p,d2)))
        dh = np.divide(2*A, d1)
        db = np.sqrt(np.subtract(np.power(d2,2), np.power(dh,2)))
        
        t1_mixed = diff_p[:,4:6] + k*np.reshape(db, (-1,1)) + kp*np.reshape(dh, (-1,1))
        t2_mixed = diff_p[:,4:6] + k*np.reshape(db, (-1,1)) - kp*np.reshape(dh, (-1,1))
        t_mixed = np.concatenate((t1_mixed, t2_mixed), axis=0)

    return t_node, t_agent, t_mixed

def getAddSolutions(readings, parents, params):

    if(not np.any(readings) or not np.any(parents)):   # if no readings collected or no parents found
        return parents
    
    rs = list(params.rs)
    r = np.tile(rs, (len(parents),1))
    structure = np.zeros((len(parents), 6))
    structure[:,0] = readings[parents[:,0],1]
    structure[:,1] = readings[parents[:,1],1]
    structure[:,2] = readings[parents[:,0],7]
    structure[:,3] = readings[parents[:,0],8]
    structure[:,4] = readings[parents[:,1],7]
    structure[:,5] = readings[parents[:,1],8]

    k = structure[:,2:4]-structure[:,4:6]
    k_prime = structure[:,4:6]-structure[:,2:4]
    d = np.linalg.norm(k, axis=1)
    k /= np.reshape(d, (-1,1))
    k_prime /= np.reshape(d, (-1,1))
    rs_oj = r[:,tuple(map(int,structure[:,1]))][0].reshape(-1,1)
    rs_oi = r[:,tuple(map(int,structure[:,0]))][0].reshape(-1,1)
    df1 = np.add(structure[:,4:6], rs_oj*k)
    df2 = np.add(structure[:,2:4], rs_oi*k_prime)
    t = np.add(df1, df2)/2
    return t



def filterSolutions(solutions, readings, rs):

    if(not np.any(solutions)):   # if no readings collected
        return solutions
    points = np.c_[np.multiply(readings[:,5],np.cos(readings[:,6])), np.multiply(readings[:,5],np.sin(readings[:,6]))] 
    dist = distance_matrix(points, solutions)

    rs = np.tile(rs, (len(dist),1))
    rs = rs[:, tuple(map(int,readings[:,1]))][0]
    rs = np.broadcast_to(rs.reshape(-1,1), (len(rs), len(solutions)))
    mask = np.all(dist>rs, axis=0)
    return solutions[mask, :]

def cartesian2polar(data):
    
    if(not np.any(data)):   # if no readings collected
        return data
    d = np.linalg.norm(data, axis=1)
    phi = np.arctan2(data[:,1], data[:,0])%(2*np.pi)
    return np.c_[d,phi]

def sortByDistance(data):
    if(not np.any(data)):   # if no readings collected
        return np.asarray([[],[]]).T
    return data[data[:,0].argsort(),:]

# N.B. readings not filtered
def computeTarget(node, readings, policy):
    
    params = GlobalConfig.cfg().params

    f_readings = readings[np.logical_not(np.logical_and(readings[:,1]==1, readings[:,2]!=2))] # filter readings for only nodes or agents bound
        
    x = np.multiply(f_readings[:,5],np.cos(f_readings[:,6]))    # Cartesian coordinates conversion: x coord
    y = np.multiply(f_readings[:,5],np.sin(f_readings[:,6]))    # Cartesian coordinates conversion: y coord
    f_readings = np.c_[f_readings, x, y]                        

    boundary = f_readings[f_readings[:,3]==0]   # only readings of nodes or bound agents of level 0
    
    rs_new = np.tile(list(params.rs), (len(boundary),1))
    indices_parents = getPairParents(boundary, rs_new, False) # find parents for traditional solutions
    indices_add_parents = getPairParents(boundary, rs_new, True) # find parents for additional solutions

    t_node, t_agent, t_mixed = getSolutions(boundary, indices_parents, params)  #compute solutions
    t_add = getAddSolutions(boundary, indices_add_parents, params)  # compute additional solutions

    t_mixed = filterSolutions(t_mixed, f_readings, list(params.rs))  # filter solutions
    t_node = filterSolutions(t_node, f_readings, list(params.rs))
    t_agent = filterSolutions(t_agent, f_readings, list(params.rs))
    t_add = filterSolutions(t_add, f_readings, list(params.rs))

    t_mixed = cartesian2polar(t_mixed)  # conversion to d, phi
    t_node = cartesian2polar(t_node)
    t_agent = cartesian2polar(t_agent)
    t_add = cartesian2polar(t_add)
    
    t_mixed = sortByDistance(t_mixed)   #sort by distance (closest first)
    t_node = sortByDistance(t_node)
    t_agent = sortByDistance(t_agent)
    t_add = sortByDistance(t_add)

    a = np.asarray([t[0] for t in [t_node, t_agent, t_mixed, t_add] if np.any(t)])
    a = sortByDistance(a)

    if(not np.any(a)):   # if no readings collected
        return [], 0
    if(a[0,0]<params.th_approach):
        return a[0], len(a)
    
    if policy=="npma":
        t = np.vstack((t_node, t_agent, t_mixed, t_add))
    elif policy=="anpm":
        t = np.vstack((t_add, t_node, t_agent, t_mixed))
    elif policy=="pmna":
        t = np.vstack((t_agent, t_mixed, t_node, t_add))
    elif policy=="closest":
        t = np.vstack((t_node, t_agent, t_mixed, t_add))
        t = sortByDistance(t)
    else:   #random
        i = [t_node, t_agent, t_mixed, t_add]
        random.shuffle(i)
        t = np.vstack(i)

    return t[0], len(t)

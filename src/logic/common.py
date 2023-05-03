import numba as nb
import numpy as np
import dask.dataframe as dd
import math
import scipy.spatial.distance
import os

from src.simulator import GlobalConfig

def get_move(dt):
    def move(args):
        x_k, y_k = args[0], args[1]
        theta_k = args[2]
        v_k = args[3]
        w_k = args[4]

        if abs(w_k) <= 0.0175:
            #Runge-Kutta with w=0
            theta_k1 = theta_k
            x_k1 = x_k + v_k*dt*math.cos(theta_k)
            y_k1 = y_k + v_k*dt*math.sin(theta_k)
        else:
            #Exact integration
            theta_k1 = (theta_k + w_k*dt) % (2*math.pi)
            x_k1 = x_k + (v_k/w_k)*(math.sin(theta_k1) - math.sin(theta_k))
            y_k1 = y_k - (v_k/w_k)*(math.cos(theta_k1) - math.cos(theta_k)) 
        
        return x_k1, y_k1, theta_k1
    return move

def range_and_bearing(objects, global_memory):
    dists = scipy.spatial.distance.pdist(objects[["x", "y"]], "euclidean")
    dists = scipy.spatial.distance.squareform(dists)
    faulty = objects["fault"] != 0
    dists[:, faulty] = float("inf")
    np.fill_diagonal(dists, float("inf"))

    def rnb_fn(object):
        object_neighbours = np.argwhere(dists[object.id, :] < object.communication_radius)[:, 0]
        rnb = objects.loc[object_neighbours]

        if len(rnb) == 0: return np.array([])
        
        point = np.array([object.x, object.y]).T
        phi = object.heading
        rnb_points = np.column_stack((rnb.x, rnb.y))

        rotation_matrix = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])

        shifted_rnb_points = rnb_points - point
        rotated_rnb_points = np.atleast_2d(np.dot(shifted_rnb_points, rotation_matrix))

        r = np.sqrt(rotated_rnb_points[:, 0]**2 + rotated_rnb_points[:, 1]**2)
        theta = (np.arctan2(rotated_rnb_points[:, 1], rotated_rnb_points[:, 0])) % (2*math.pi)

        return np.stack((
            rnb.id,
            rnb.type == "agent",
            rnb.state,
            rnb.level,
            rnb.gossip,
            r,
            theta
        )).T
    
    

    objects["range_and_bearing"] = objects.apply(rnb_fn, axis=1)
    return objects


import numpy as np
import random
import pygame
import math
from src.logic.controller.render import render_nodes, render_agents
from src.logic.controller.utils import \
    heardHole, \
    updateLevel, \
    followMin, \
    target2control, \
    computeTarget, \
    perturbate

from src.simulator import GlobalConfig
from src.simulator import Hook

class Msg(Hook):
    def __init__(self, cfg):
        pass

    # packet_sent: list of dictionaries whose fields are: destination (id of the recepient); message (message)
    # packet_received: 
    def exchangeMsg(self, objects):
        
        objects["packet_received"]= [[] for j in range(len(objects))]  # clean all received packets
        for i in range(len(objects)):
                if len(objects.at[i, "packet_sent"])!=0:    # if the node i has to send at least a packet
                    msgs = objects.at[i, "packet_sent"]
                    for packet in msgs: # to account for multiple messages to send
                        dest_id = packet["destination"]
                        msg = objects.at[dest_id, "packet_received"]    # to account for packets received from multiple senders
                        msg.append({"sender": i, "message":packet["message"]})
                        objects.at[dest_id, "packet_received"] = msg
                    objects.at[i, "packet_sent"] = []   # msgs sent, clear request



def events_fn(*, objects, events, state):
    ret = dict()
    for event in events:
        if event.type == pygame.QUIT:
            ret["quit"] = True
        elif event.type == pygame.KEYDOWN:
            key = pygame.key.name(event.key)
            if key == "i":
                if "display_ids" not in state: state["display_ids"] = True
                else: state["display_ids"] = not state["display_ids"]
            if key == "d":
                fname = input("Enter dump filename (dump.pkl): ")
                if fname == "": fname = "dump.pkl"
                objects.to_pickle(fname)
                print(f"Saved {fname}")
            if key == "a":
                if "display_active" not in state: state["display_active"] = False
                else: state["display_active"] = not state["display_active"]

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            nearest = objects.iloc[0]
            mindist = math.sqrt((nearest.x - pos[0])**2 + (nearest.y - pos[1])**2)
            for idx, object in objects.iterrows():
                dist = math.sqrt((object.x - pos[0])**2 + (object.y - pos[1])**2)
                if dist < mindist:
                    nearest = object
                    mindist = dist
            
            print(nearest)
            print("READINGS")
            print(nearest.range_and_bearing)
    
    return ret

def render_fn(*, window, objects, state):
    nodes = objects[objects["type"] == "node"]
    agents = objects[objects["type"] == "agent"]

    render_nodes(nodes=nodes, window=window, state=state)
    render_agents(agents=agents, window=window, state=state)


def agent_logic(node):

    v = 0
    w = 0
    readings = node.range_and_bearing
    if len(node.range_and_bearing) == 0: return node
    
    params = GlobalConfig.cfg().params

    p_break = 1-np.power(1-params.fault_perc, 1/GlobalConfig.cfg().max_ticks)

    if random.random() < p_break:
        node.type = "agent_faulty"
        node.fault = True
        return node

    readings = perturbate(readings, params)

    if node.state == 0:    # INACTIVE
        if(heardHole(readings)):    # if holes are detected
            node.state = 1

    elif node.state == 1:    # ACTIVE
        if(not heardHole(readings)):    # if holes are no longer detected
            node.state = 0
        else:
            border_readings = readings[np.logical_not(np.logical_and(readings[:,1]==1, readings[:,2]!=2))] # filter readings for only nodes or agents bound
            lev_zero = border_readings[border_readings[:,3]==0]  # filter readings for only level=0
            if(len(lev_zero)>1): # if enough readings for computing boundary solutions
                t, nt = computeTarget(node, readings, params.policy)
                if(len(t)!=0):  # a solution was found
                    if(t[0]<=params.th_bound):  # if close enough to target, then switch state to bound
                        node.state = 2
                    else:
                        v, w = target2control(t[0], t[1], node)
                    node.v = v
                    node.w = w
                    return node

            # No feasible solution found
            d_min, angle_min = followMin(border_readings)
            v, w = target2control(d_min, angle_min, node)

    elif node.state == 2:    # BOUND
        if(not heardHole(readings)):    # if holes are no longer detected
            node.state = 0
        else:
            node.level = updateLevel(node)

    node.v = v
    node.w = w
    return node




def node_logic(node):
    if len(node.range_and_bearing) == 0: return node
    if not node.fault:
        node.level = updateLevel(node)
    return node

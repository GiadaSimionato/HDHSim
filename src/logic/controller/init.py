import math
import pandas as pd
import numpy as np

def init_nodes(cfg, nodes, init_id=0):
    ret = []
    for i, node in enumerate(nodes):
        ret.append(dict(
            id=init_id+i,
            x=node.x,
            y=node.y,
            heading=node.heading,
            fault=bool(node.fault),
            boundary=bool(node.boundary),
            level=-1,
            state=-1,
            gossip=1,
            communication_radius=cfg.params.rc[0],
            range_and_bearing=[],
            packet_sent = [],
            packet_received = []
        ))
    return pd.DataFrame(ret)

def init_agents(cfg, init_id=0):
    ret = []
    agent_id = 0
    for nest_i in range(0, cfg.init.agents.nests):
        nest_x = cfg.init.agents.margin[0] + np.random.uniform(0, cfg.world.size[0] - cfg.init.agents.margin[0])
        nest_y = cfg.init.agents.margin[1] + np.random.uniform(0, cfg.world.size[1] - cfg.init.agents.margin[1])

        for drone_i in range(0, cfg.init.agents.nest.agents):
            x = np.random.normal(loc=nest_x, scale=cfg.init.agents.nest.std)
            y = np.random.normal(loc=nest_y, scale=cfg.init.agents.nest.std)

            ret.append(dict(
                id=init_id+agent_id,
                x=x,
                y=y,
                v=0.0,
                w=0.0,
                boundary = False,   # if agent is on the boundary of the area (do not change)
                heading=np.random.uniform(0, 2*math.pi),
                fault=False,
                level=-1,
                state=0,
                gossip=1,
                communication_radius=cfg.params.rc[1],
                range_and_bearing=[],
                packet_sent = [],
                packet_received = []
            ))
            agent_id += 1

    return pd.DataFrame(ret)


def scale_params(cfg, scale_factor):
    scale_factor = float(scale_factor)

    cfg.params.rs[0] /= scale_factor
    cfg.params.rs[1] /= scale_factor

    cfg.params.rc[0] /= scale_factor
    cfg.params.rc[1] /= scale_factor

    cfg.params.driving_speed_cruise /= scale_factor
    cfg.params.errors.rab_dist_more *= scale_factor
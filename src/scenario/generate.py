import hydra
import tempfile
import matplotlib.pyplot as plt
from shapely import geometry
from src.scenario.utils import *

def buildNetwork(cfg):

    area = geometry.Polygon([[cfg.network.x_limits[1], cfg.network.y_limits[0]], [cfg.network.x_limits[0], cfg.network.y_limits[0]], [cfg.network.x_limits[0], cfg.network.y_limits[1]], [cfg.network.x_limits[1], cfg.network.y_limits[1]]])
    n_placed = 0
    points = []

    while n_placed < cfg.network.n_nodes:
        i = 0
        while i < cfg.network.n_iter:
            pnt = samplePoint(area)
            if validPoint(pnt, points, cfg.network.rs_nodes, cfg.network.epsilon):
                points.append(pnt)
                n_placed += 1
                break
            else:
                i += 1
        if i>=cfg.network.n_iter:
            break

    # --- Start Plot ---        
    fig, ax = plt.subplots(figsize=(10, 10))
    for point in points:
        ax.add_patch(plt.Circle((point.x, point.y), cfg.network.rs_nodes, color='g', fill=False))
    x_points = [point.x for point in points]
    y_points = [point.y for point in points]
    plt.scatter(x_points, y_points, s=1, c='b')

    x,y = area.exterior.xy
    plt.plot(x,y)
    # --- End Plot --- 
    
    union = getCoverage(area, points, cfg.network.rs_nodes)

    # --- Start Plot --- 
    x,y = union.exterior.xy
    plt.plot(x,y, color="red", lw=1)

    for inner in union.interiors:
        xi, yi = zip(*inner.coords[:])
        ax.plot(xi, yi, color="red", lw=1)
    # --- End Plot --- 

    intersection = area.intersection(union)

    # --- Start Plot --- 
    x,y = intersection.exterior.xy
    plt.plot(x,y, color="yellow", lw=1)

    for inner in intersection.interiors:
        xi, yi = zip(*inner.coords[:])
        ax.plot(xi, yi, color="yellow", lw=1)
    # --- End Plot --- 
    
    for interior in intersection.interiors:
        interior = Polygon(interior)
        if interior.exterior.length >= cfg.network.th_inner:
            c = interior.centroid
            points.append(c)
            # --- Start Plot ---
            plt.plot([c.x],[c.y], marker = '.', markeredgecolor="black", markerfacecolor="black")
            ax.add_patch(plt.Circle((c.x, c.y), cfg.network.rs_nodes, color='g', fill=False))
            # --- End Plot ---    

    # --- Start Plot ---
    plt.gca().set_aspect('equal', adjustable='box')
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(f.name, dpi=300)
    # --- End Plot ---

    return area, points, f.name

def breakNodes(nodes, ntb, rs):
    
    for node in nodes:  
        if node.fault:  # propagate fault to neighbours
            for ng in node.neighbours:
                ng.ng_to_fault = True
        if node.boundary:   # propagate boundary to neighbours
            for ng in node.neighbours:
                ng.ng_to_boundary = True

    nodes_new = [node for node in nodes if not node.fault and not node.ng_to_fault and not node.boundary and not node.ng_to_boundary] # remove faulty, boundary, near to faulty or boundary nodes
    nodes_new = updateNeighbours(nodes_new)
    cc = getConnectedComponents(nodes_new, ntb)
    if len(cc)==0:
        print("Not enough space to create the hole")
        raise SystemExit
    cc = [c for c in cc if len(c)>=ntb] # filter for cc of length >=ntb
    index = random.randint(0, len(cc)-1)
    c = cc[index]   #random selection of connected component
    ids = findHole(c, ntb, rs)

    for i in ids:
        index = getId(nodes, i)
        nodes[index].fault = True
    
    return nodes

def generate(cfg):
    rs_nodes = cfg.params.rs[0]
    area, points, init_network_plot = buildNetwork(cfg)    # build the network
    #nodes = setBoundary(area, points, rs_nodes) # set boundary flag and convert to Node objects
    nodes = setBoundary(points, rs_nodes) # set boundary flag and convert to Node objects
    nodes = setNeighbours(nodes, rs_nodes)  # setNeighbours
    
    for hole_size in cfg.scenario.ntb:
        nodes = breakNodes(nodes, hole_size, rs_nodes)  # Create hole
    
    final_network_plot = renderNetwork(nodes, rs_nodes)  # Render network
    scale_factor, nodes = scaleData(
        nodes=nodes,
        margin=cfg.world.margin,
        size=cfg.world.size
    )

    return dict(
        nodes=nodes,
        scale_factor=scale_factor,
        plots=dict(
            init_network=init_network_plot,
            final_network=final_network_plot
        )
    )

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    generate(cfg)

if __name__ == "__main__":
    main()
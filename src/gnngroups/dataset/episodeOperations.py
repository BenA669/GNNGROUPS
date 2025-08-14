import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from noise import pnoise2
import time as time
from tqdm import tqdm
from gnngroups.utils import *
from .pygameAnimate import animatev2

def adjacency_to_edge_index(adj_t: torch.Tensor):
    # (node_i, node_j) for all 1-entries
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index

def get_perlin_gradient(x, y, offset, noise_scale=1.0, eps=1e-3, octaves=1, persistence=0.5, lacunarity=2.0):
    n1 = pnoise2((x + eps + offset[0]) * noise_scale,
                 (y + offset[1]) * noise_scale,
                 octaves=octaves,
                 persistence=persistence,
                 lacunarity=lacunarity)
    n2 = pnoise2((x - eps + offset[0]) * noise_scale,
                 (y + offset[1]) * noise_scale,
                 octaves=octaves,
                 persistence=persistence,
                 lacunarity=lacunarity)
    dx = (n1 - n2) / (2 * eps)

    n3 = pnoise2((x + offset[0]) * noise_scale,
                 (y + eps + offset[1]) * noise_scale,
                 octaves=octaves,
                 persistence=persistence,
                 lacunarity=lacunarity)
    n4 = pnoise2((x + offset[0]) * noise_scale,
                 (y - eps + offset[1]) * noise_scale,
                 octaves=octaves,
                 persistence=persistence,
                 lacunarity=lacunarity)
    dy = (n3 - n4) / (2 * eps)
    
    return dx, dy


def genDataset(node_amt             = dataset_cfg['nodes'],
               group_amt            = dataset_cfg['groups'],
               time_steps           = dataset_cfg['timesteps'],
               boundary             = dataset_cfg['boundary'],
               noise_strength       = dataset_cfg['noise_strength'],
               noise_scale          = dataset_cfg['noise_scale'],
               distance_threshold   = dataset_cfg['distance_threshold'],
               nhops                = dataset_cfg['hops'],
               ):
    
    positions_t_n_xy = torch.zeros((time_steps, node_amt, 2))
    offsets_n_2 = torch.rand((node_amt, 2))*boundary*2-boundary
    grad_intensity_n = torch.full((node_amt, ), 0.1)
    
    nodes_per_slice = math.ceil(math.sqrt(node_amt))
    seed = torch.linspace(-boundary, boundary, nodes_per_slice)
    inital_pos = torch.cartesian_prod(seed, seed)[:node_amt]
    positions_t_n_xy[0, :, :] = inital_pos

    for t in tqdm(range(1, time_steps)):
        for n in range(node_amt):
            x = positions_t_n_xy[t-1, n, 0]
            y = positions_t_n_xy[t-1, n, 1]
            
            grad_x, grad_y = get_perlin_gradient(
                x, y, offset=offsets_n_2[n], noise_scale=noise_scale
            )            

            new_post = [x + grad_x * noise_strength * grad_intensity_n[n], y + grad_y * noise_strength * grad_intensity_n[n]] 
            if abs(grad_x)+abs(grad_y) < 0.1 or abs(new_post[0]) > boundary or abs(new_post[1]) > boundary:
                offsets_n_2[n, 0] = random.uniform(-5000, 5000)
                offsets_n_2[n, 1] = random.uniform(-5000, 5000)
                grad_intensity_n[n] = 0.1
            new_post[0] = max(-boundary, min(new_post[0], boundary))
            new_post[1] = max(-boundary, min(new_post[1], boundary))

            positions_t_n_xy[t, n, :] = torch.tensor(new_post)
            grad_intensity_n[n] += 0.05
            grad_intensity_n[n] = min(1, grad_intensity_n[n])
        
    return positions_t_n_xy

def genBulkDataset():
    time_steps = dataset_cfg["timesteps"]
    node_amt = dataset_cfg["nodes"]
    samples = training_cfg["samples"]

    dataset_pos = torch.empty(samples, time_steps, node_amt, 2)

    save_file = dataset_cfg["dir_path"] + dataset_cfg["dataset_name"]
    print("Saving positions to {}".format(save_file + "_pos.pt"))

    for i in tqdm(range(samples)):
        positions_t_n_xy = genDataset()
        dataset_pos[i] = positions_t_n_xy
    
    torch.save(dataset_pos, save_file + "_pos.pt")
    print("Saved positions to {}".format(save_file + "_pos.pt"))
    return


def genAnchors(positions_t_n_xy,
               anchor_ratio         = dataset_cfg['anchor_node_ratio'],
               distance_threshold   = dataset_cfg['distance_threshold']):
    time_steps, node_amt, _ = positions_t_n_xy.shape

    # Select anchor nodes
    anchor_amt = int(np.floor(node_amt * anchor_ratio))
    anchor_indices_n = torch.randperm(node_amt)[:anchor_amt]
    # print(f"anchor indices: {anchor_indices_n}")

    # Make anchors still
    new_positions_t_n_xy = positions_t_n_xy.clone()
    new_positions_t_n_xy[:, anchor_indices_n, :] = new_positions_t_n_xy[0, anchor_indices_n, :]

    # Initialize X Distance Matrix
    # Setup anchor-agent and anchor-anchor distances of X(N, N)
    X = torch.cdist(new_positions_t_n_xy, new_positions_t_n_xy)     # Get full distance matrix
    X_temp = X.clone()

    mask_anchor = torch.zeros((node_amt, node_amt), dtype=torch.bool)   # Mask out 
    mask_anchor[anchor_indices_n, :] = True
    mask_anchor[:, anchor_indices_n] = True
    X[:, ~mask_anchor] = 0

    A_t_n_n = X_temp < distance_threshold # Create Adj Matrix
    Xhat_t_n_n = A_t_n_n * X # Create Xhat

    anchor_pos_t_n_xy = torch.zeros(time_steps, node_amt, 2)
    anchor_pos_t_n_xy[:, anchor_indices_n, :] = new_positions_t_n_xy[:, anchor_indices_n, :]

    return new_positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, anchor_indices_n

def makeEpisode():
    anchor_ratio = dataset_cfg["anchor_node_ratio"]
    distance_threshold = dataset_cfg["distance_threshold"]

    positions_t_n_xy = genDataset()

    new_positions_t_n_xy, \
    Xhat_t_n_n, \
    A_t_n_n, \
    anchor_pos_t_n_xy, \
    anchor_indices_n = genAnchors(positions_t_n_xy, anchor_ratio, distance_threshold)

    animatev2(new_positions_t_n_xy, A_t_n_n, anchor_indices_n)

    return
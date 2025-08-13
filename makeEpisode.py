import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from noise import pnoise2
from animate import plot_faster
from pygameAnimate import animate, animatev2
import time as time
from configReader import read_config
from tqdm import tqdm

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

model_cfg, dataset_cfg, training_cfg = read_config("config.ini")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # adjacency_t_n_n  = torch.zeros((time_steps, node_amt, node_amt))
    offsets_n_2 = torch.rand((node_amt, 2))*boundary*2-boundary
    grad_intensity_n = torch.full((node_amt, ), 0.1)
    
    nodes_per_slice = math.ceil(math.sqrt(node_amt))
    seed = torch.linspace(-boundary, boundary, nodes_per_slice)
    inital_pos = torch.cartesian_prod(seed, seed)[:node_amt]
    positions_t_n_xy[0, :, :] = inital_pos
    # adjacency_t_n_n[0, :, :] = torch.cdist(positions_t_n_xy[0, :, :].unsqueeze(0), 
                                    #  positions_t_n_xy[0, :, :].unsqueeze(0)).squeeze(0) < distance_threshold

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
        
        # adjacency_t_n_n[t] = torch.cdist(positions_t_n_xy[t, :, :].unsqueeze(0), 
        #                                  positions_t_n_xy[t, :, :].unsqueeze(0)).squeeze(0) < distance_threshold
        

    # Zero out self connections
    # mask = torch.eye(node_amt, dtype=torch.bool).repeat(time_steps, 1, 1)
    # adjacency_t_n_n[mask] = 0 

    # n_hop_adjacency = torch.linalg.matriux_power(adjacency_t_n_n, nhops)

    # n_hop_adjacency_t_h_n_n = torch.zeros((time_steps, nhops, node_amt, node_amt))
    # for hop in range(nhops):
    #     if hop == 0:
    #         n_hop_adjacency_t_h_n_n[:, hop] = adjacency_t_n_n
    #     else:
    #         n_hop_adjacency_t_h_n_n[:, hop] = torch.linalg.matrix_power(adjacency_t_n_n, hop + 1)

    # return positions_t_n_xy, n_hop_adjacency_t_h_n_n
    return positions_t_n_xy

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

    return new_positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy


def getSubnet(positions_t_n_xy, n_hop_adjacency_t_h_n_n, nhops, ego_idx, timestep):
    adjacency_n_n = n_hop_adjacency_t_h_n_n[timestep, 0]
    positions_n_xy = positions_t_n_xy[timestep]

    # Get masks for core and all indices
    all_indices_n_mask =  (n_hop_adjacency_t_h_n_n[timestep, :nhops, ego_idx, :] > 0).any(dim=0)
    all_indices_n_mask[ego_idx] = True
    core_indices_n_mask = n_hop_adjacency_t_h_n_n[timestep, :nhops-1, ego_idx, :].any(dim=0) > 0
    core_indices_n_mask[ego_idx] = True
    core_indices_n_mask = core_indices_n_mask[all_indices_n_mask] 

    # Keep nodes in nhop then mask out adj rows for edge node rows
    adj_subnet_sn_sn = adjacency_n_n[all_indices_n_mask][:, all_indices_n_mask]
    adj_subnet_sn_sn[~core_indices_n_mask][:, ~core_indices_n_mask] = 0

    # Restore self loops
    adj_subnet_sn_sn[torch.eye(adj_subnet_sn_sn.size(0)) == 1] = 1
    pos_subnet_sn_xy = positions_n_xy[all_indices_n_mask]

    mask_indices_n_global = torch.where(all_indices_n_mask)[0]

    return pos_subnet_sn_xy, adj_subnet_sn_sn, mask_indices_n_global

    
def getSubnetBatched(
        positions_b_t_n_xy,
        n_hop_adjacency_b_t_h_n_n,
        nhops,
        ego_idx,
        timestep
):
    B, T, N, _ = positions_b_t_n_xy.shape

    adjacency_b_n_n = n_hop_adjacency_b_t_h_n_n[:, timestep, :, :, :]
    positions_b_n_xy = positions_b_t_n_xy[:, timestep, :]

    return False

def pingpong(num_nodes, ping_i=None, pong_i=None):
    if ping_i is None:
        ping_i = torch.randint(0, num_nodes, (1,)).item()
    
    if pong_i is None:
        while True:
            pong_i = torch.randint(0, num_nodes, (1,)).item()
            if pong_i != ping_i:
                break


def makeDatasetDynamicPerlin(
    node_amt=400,
    group_amt=4,
    std_dev=1,
    time_steps=20,
    distance_threshold=2,
    noise_scale=1.0,      # scales the noise coordinate (affects frequency)
    noise_strength=0.1,   # multiplier for the noise gradient (step size)
    tilt_strength=0.05,   # added constant bias (tilt) for each group
    octaves=1,
    persistence=0.5,
    lacunarity=2.0,
    boundary=4,
    perlin_offset=0.05,
    mixed=False,
    rng=None
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.initial_seed())

    # Assign nodes to groups and set initial positions ---
    # if mixed:
    #     rand_vals = torch.rand(node_amt, generator=rng, device=device)
    #     group_amt_r = torch.randint(2, group_amt+1, (1,), generator=rng, device=device).item()
    #     group_amt = group_amt_r
    #     group_edges = torch.arange(1, group_amt_r + 1, device=device) * (1.0 / group_amt)
    #     groups = torch.bucketize(rand_vals, group_edges)  # each in [0, group_amt - 1]
    # else:
    #     print("passed")
    #     rand_vals = torch.rand(node_amt, generator=rng, device=device)
    #     group_edges = torch.arange(1, group_amt + 1, device=device) * (1.0 / group_amt)
    #     groups = torch.bucketize(rand_vals, group_edges)  # each in [0, group_amt - 1]
    rand_vals = torch.rand(node_amt, generator=rng, device=device)
    group_edges = torch.arange(1, group_amt + 1, device=device) * (1.0 / group_amt)
    groups = torch.bucketize(rand_vals, group_edges)  # each in [0, group_amt - 1]


    # Use a random seed (group center) for each group.
    last_seen_seeds = torch.rand((group_amt, 2), device=device)  # shape: (group_amt, 2)
    nodes = torch.zeros((node_amt, 3), device=device)             # shape: (node_amt, 3)
    nodes[:, 2] = groups  # store group id in third column

    
    
    for i in range(node_amt):
        g = int(groups[i].item())
        # Create an initial position by sampling from a normal distribution centered at the group's current seed.
        point = torch.normal(mean=last_seen_seeds[g], std=std_dev, generator=rng)
        last_seen_seeds[g] = point  # update the group's seed
        nodes[i, 0] = point[0]
        nodes[i, 1] = point[1]

    all_positions = torch.zeros((time_steps, node_amt, 3), device=device)
    all_positions[0] = nodes

    # Set up Perlin noise parameters, group tilts, and group speeds
    # Each group gets its own noise offset so that its noise field is different.
    # Each group is assigned a constant "tilt" direction.
    # Assign each group a speed multiplier: 0.5 with 50% chance, or 1.0 with 50% chance.
    group_noise_offsets = []
    group_tilts = []
    group_speeds = []  # New: store speed multipliers per group.
    for g in range(group_amt):
        offset_x = random.uniform(0, 100)
        offset_y = random.uniform(0, 100)
        group_noise_offsets.append((offset_x, offset_y))
        # Create a random tilt vector (unit vector) for the group.
        angle = random.uniform(0, 2 * np.pi)
        tilt_vector = (np.cos(angle), np.sin(angle))
        group_tilts.append(tilt_vector)
        speed = 0.3 if random.random() < 0.5 else 1.0
        group_speeds.append(speed)

    eps = 1e-3  # small epsilon for finite difference gradient approximation

    adjacency_dynamic = []
    group_edge_amount = torch.zeros(group_amt)

    for t in range(time_steps):
        if t != 0:
            prev_positions = all_positions[t - 1].clone()
            new_positions = prev_positions.clone()
            for i in range(node_amt):
                # Get current x, y and the group id of this node.
                x = prev_positions[i, 0].item()
                y = prev_positions[i, 1].item()
                group = int(prev_positions[i, 2].item())
                offset = group_noise_offsets[group]
                tilt_vector = group_tilts[group]
                speed_multiplier = group_speeds[group]


                # Compute the gradient of the noise field at (x, y)
                grad_x, grad_y = get_perlin_gradient(
                    x, y, offset, noise_scale, eps, octaves, persistence, lacunarity
                )

                if speed_multiplier > 0.5:
                    noise_strength_mod = noise_strength/1.5
                else:
                    noise_strength_mod = noise_strength
                # Combine noise gradient with the constant tilt.
                step_x = noise_strength_mod * grad_x + tilt_strength * tilt_vector[0]
                step_y = noise_strength_mod * grad_y + tilt_strength * tilt_vector[1]

                # Multiply the step by the group's speed multiplier.
                step_x *= speed_multiplier
                step_y *= speed_multiplier

                new_positions[i, 0] = prev_positions[i, 0] + step_x
                new_positions[i, 1] = prev_positions[i, 1] + step_y
                # The group id (stored in column 2) remains unchanged.
            for g in range(group_amt):
                group_indices = (new_positions[:, 2] == g)
                if group_indices.sum() > 0:
                    group_positions = new_positions[group_indices, :2]
                    mean_pos = group_positions.mean(dim=0)
                    mean_x, mean_y = mean_pos[0].item(), mean_pos[1].item()
                    tilt_x, tilt_y = group_tilts[g]
                    if mean_x > boundary:
                        tilt_x = -abs(tilt_x)
                    elif mean_x < -boundary:
                        tilt_x = abs(tilt_x)
                    if mean_y > boundary:
                        tilt_y = -abs(tilt_y)
                    elif mean_y < -boundary:
                        tilt_y = abs(tilt_y)
                    group_tilts[g] = (tilt_x, tilt_y)
                group_noise_offsets[g] = (group_noise_offsets[g][0]+perlin_offset, group_noise_offsets[g][1]+perlin_offset)
            all_positions[t] = new_positions


        positions_t = all_positions[t, :, :2]  # shape: (node_amt, 2)
        # Compute pairwise distances.
        dist_mat = torch.cdist(positions_t.unsqueeze(0), positions_t.unsqueeze(0)).squeeze(0)
        # Build dynamic adjacency from distance threshold.
        dist_adj = (dist_mat < distance_threshold).to(torch.int8)
        dist_adj.fill_diagonal_(0)
        dist_adj = torch.triu(dist_adj, diagonal=1) + torch.triu(dist_adj, diagonal=1).T
        # Combine static and dynamic connections (logical OR).
        adjacency_dynamic.append(dist_adj)

        # Group inter edges amount
        for g in range(group_amt):
            group_indicies = (all_positions[t, :, 2] == g)
            non_group_indicies = (all_positions[t, :, 2] != g)
            group_edge_amount[g] = dist_adj[group_indicies][:, non_group_indicies].flatten().sum()

        # print(group_edge_amount)

        for g in range(group_amt):
            group_indicies = (all_positions[t, :, 2] == g)
            non_group_indicies = (all_positions[t, :, 2] != g)
            group_edge_amount[g] = dist_adj[group_indicies][:, non_group_indicies].flatten().sum()        

    adjacency_dynamic = torch.stack(adjacency_dynamic, dim=0)

    all_positions_cpu = all_positions.cpu()
    adjacency_dynamic_cpu = adjacency_dynamic.cpu()

    edge_indices = []
    for t in range(time_steps):
        adj_t = adjacency_dynamic_cpu[t]
        edge_index_t = adjacency_to_edge_index(adj_t)
        edge_index_t = edge_index_t.to(device)
        edge_indices.append(edge_index_t)

    return all_positions_cpu, adjacency_dynamic_cpu, edge_indices, group_amt

def getEgo(all_positions_cpu, adjacency_dynamic_cpu, min_groups=3, hop=2, union=True):

    time_steps, total_nodes, _ = all_positions_cpu.shape

    union_adj = (adjacency_dynamic_cpu.any(dim=0) > 0)

    def get_hop_network_and_distances(ego_idx, max_hop, union=True):
        distances = torch.full((total_nodes,), -1, dtype=torch.int)
        distances[ego_idx] = 0
        frontier = [ego_idx]
        current_distance = 0

        if not(union):
            distances_t = torch.full((time_steps, total_nodes,), -1, dtype=torch.int)
            for time in range(time_steps):
                # distances = torch.full((total_nodes,), -1, dtype=torch.int)
                distances_t[time, ego_idx] = 0
                frontier = [ego_idx]
                current_distance = 0
                while frontier and current_distance < max_hop:
                    next_frontier = []
                    for node in frontier:
                        # Find neighbors of node in the union graph.
                        neighbors = torch.nonzero(adjacency_dynamic_cpu[time, node], as_tuple=False).flatten().tolist()
                        for nb in neighbors:
                            if distances_t[time, nb] == -1:  # not visited yet
                                distances_t[time, nb] = current_distance + 1
                                next_frontier.append(nb)
                    frontier = next_frontier
                    current_distance += 1
            return distances_t

        while frontier and current_distance < max_hop:
            next_frontier = []
            for node in frontier:
                # Find neighbors of node in the union graph.
                neighbors = torch.nonzero(union_adj[node], as_tuple=False).flatten().tolist()
                for nb in neighbors:
                    if distances[nb] == -1:  # not visited yet
                        distances[nb] = current_distance + 1
                        next_frontier.append(nb)
            frontier = next_frontier
            current_distance += 1
        return distances

    candidate_indices = list(range(total_nodes))
    random.shuffle(candidate_indices)
    chosen_idx = None

    for candidate in candidate_indices:
        distances = get_hop_network_and_distances(candidate, hop, union=True)
        reachable = distances >= 0  # nodes reached within 'hop' hops
        neighbor_indices = torch.nonzero(reachable, as_tuple=False).flatten()
        neighbor_groups = all_positions_cpu[0, neighbor_indices, 2]
        unique_groups = torch.unique(neighbor_groups)
        if unique_groups.numel() >= min_groups:
            chosen_idx = candidate
            break

    if chosen_idx is None:
        chosen_idx = random.randint(0, total_nodes - 1)
    
    ego_idx = chosen_idx
    distances = get_hop_network_and_distances(ego_idx, hop, union=union)
    reachable = distances >= 0  # static mask for nodes in the ego network (union over time)

    outer_mask = (distances == hop)
    prune_mask = outer_mask.unsqueeze(1) & outer_mask.unsqueeze(2)
    pruned_adj = adjacency_dynamic_cpu.clone()
    pruned_adj[prune_mask] = 0 # Prune connections among outer-layer nodes.

    return ego_idx, pruned_adj, reachable



if __name__ == '__main__':

    # pos, adj = genDataset()
    pos = genDataset()
    pos2, Xhat_t_n_n, adjacency_t_n_n, anchor_indices_n = genAnchors(pos)

    animatev2(pos2, adjacency_t_n_n, anchor_indices_n)
    exit()

    # plot_faster(pos, adj)
    animate(pos, adjacency_t_n_n, 
            num_timesteps=pos.shape[0], 
            num_nodes=pos.shape[1], 
            scale=50, 
            nhops=2)
    # model_cfg, dataset_cfg, training_cfg = read_config("config.ini")
    
    # time_steps = dataset_cfg["timesteps"]
    # group_amt = dataset_cfg["groups"]
    # node_amt = dataset_cfg["nodes"]

    # distance_threshold = dataset_cfg["distance_threshold"]
    # noise_scale =dataset_cfg["noise_scale"]      # frequency of the noise
    # noise_strength = dataset_cfg["noise_strength"]     # influence of the noise gradient
    # tilt_strength = dataset_cfg["tilt_strength"]   # constant bias per group
    # boundary = dataset_cfg["boundary"]

    # hops = dataset_cfg["hops"]
    # min_groups = dataset_cfg["min_groups"]

    # samples = dataset_cfg["samples"]

    # perlin_offset_amt = dataset_cfg["perlin_offset_amt"]
    
    # all_positions_cpu, adjacency_dynamic_cpu, edge_indices, groups_r = makeDatasetDynamicPerlin(
    #     node_amt=node_amt,
    #     group_amt=group_amt,
    #     time_steps=time_steps,
    #     distance_threshold=distance_threshold,
    #     noise_scale=noise_scale,
    #     noise_strength=noise_strength,
    #     tilt_strength=tilt_strength,
    #     boundary=boundary,
    #     perlin_offset=perlin_offset_amt,
    #     mixed=False
        
    # )

    # # ego_index, ego_positions, ego_adjacency, ego_edge_indices, EgoMask = getEgo(all_positions_cpu, adjacency_dynamic_cpu, hop=2, union=False)
    # ego_index, pruned_adj, reachable = getEgo(all_positions_cpu, adjacency_dynamic_cpu, hop=hops, union=False, min_groups=groups_r)

    # if training_cfg["demo"]:
    #     plot_faster(all_positions_cpu, adjacency_dynamic_cpu, ego_idx=ego_index, ego_mask=reachable)

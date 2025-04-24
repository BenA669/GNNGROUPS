import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from noise import pnoise2
from animate import plot_faster
import time as time
import configparser

def adjacency_to_edge_index(adj_t: torch.Tensor):
    # (node_i, node_j) for all 1-entries
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index

def get_perlin_gradient(x, y, offset, noise_scale, eps, octaves, persistence, lacunarity):
    """
    Compute the gradient of a Perlin noise function at (x,y) with a given offset,
    using central finite differences.
    """
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

    # --- 1. Assign nodes to groups and set initial positions ---
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


    # Use a random “seed” (group center) for each group.
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

    # --- 2. Set up Perlin noise parameters, group tilts, and group speeds ---
    # Each group gets its own noise offset so that its noise field is different.
    # In addition, each group is assigned a constant "tilt" direction.
    # Also, assign each group a speed multiplier: 0.5 with 50% chance, or 1.0 with 50% chance.
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

    # --- 3. Update node positions over time based on noise gradient, tilt, and group speed ---
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

        # --- 5. For each time step, add dynamic (distance–based) connections ---

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

        # Group inter edges amount: count edges that connect nodes in group g to nodes in other groups.
        for g in range(group_amt):
            group_indicies = (all_positions[t, :, 2] == g)
            non_group_indicies = (all_positions[t, :, 2] != g)
            group_edge_amount[g] = dist_adj[group_indicies][:, non_group_indicies].flatten().sum()        

    adjacency_dynamic = torch.stack(adjacency_dynamic, dim=0)

    # Move to CPU for convenience.
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
    """
    Select an ego node whose dynamic network (up to `hop` hops) spans at least `min_groups`
    distinct groups, and then return an ego network that prunes connections among nodes that are
    exclusively in the outermost (max hop) layer. Also returns an EgoMask which is a boolean
    tensor (time_steps, total_nodes) indicating for each time step which nodes (of the original
    total nodes) are part of the ego network (i.e. have a direct dynamic connection to the ego).

    Parameters:
        all_positions_cpu (torch.Tensor): Tensor of shape (time_steps, node_amt, 3)
            containing node positions (and group id in column 2) for each time step.
        adjacency_dynamic_cpu (torch.Tensor): Tensor of shape (time_steps, node_amt, node_amt)
            containing the dynamic (time-varying) adjacency matrices.
        min_groups (int): Minimum number of distinct groups (including the ego's own group)
            required in the ego network.
        hop (int): Number of hops to expand from the ego node.
    
    Returns:
        ego_idx (int): The index of the chosen ego node.
        ego_positions (torch.Tensor): Positions tensor for the ego network,
            of shape (time_steps, n_ego_nodes, 3).
        ego_adjacency (torch.Tensor): Modified dynamic adjacency tensor for the ego network,
            of shape (time_steps, n_ego_nodes, n_ego_nodes). For N-hop networks
        ego_edge_indices (list): List of edge indices for each time step of the ego network.
        EgoMask (torch.Tensor): Boolean mask (time_steps, total_nodes) indicating for each time
            step which nodes (in the full graph) are in the ego network based on dynamic connectivity.
    """


    time_steps, total_nodes, _ = all_positions_cpu.shape

    # Build a static union graph over time (True if an edge ever exists).
    union_adj = (adjacency_dynamic_cpu.any(dim=0) > 0)

    # Helper: Compute hop distances from the ego using BFS up to a maximum of 'hop'
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

    # --- Candidate Selection ---
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

    config = configparser.ConfigParser()
    config.read('config.ini')
    
    time_steps = int(config["dataset"]["timesteps"])
    group_amt = int(config["dataset"]["groups"])
    node_amt = int(config["dataset"]["nodes"])

    distance_threshold = int(config["dataset"]["distance_threshold"])
    noise_scale = float(config["dataset"]["noise_scale"])      # frequency of the noise
    noise_strength = float(config["dataset"]["noise_strength"])      # influence of the noise gradient
    tilt_strength = float(config["dataset"]["tilt_strength"])     # constant bias per group
    boundary = int(config["dataset"]["boundary"])

    hops = int(config["dataset"]["hops"])
    min_groups = int(config["dataset"]["min_groups"])

    samples = int(config["dataset"]["samples"])

    perlin_offset_amt = float(config["dataset"]["perlin_offset_amt"])
    
    all_positions_cpu, adjacency_dynamic_cpu, edge_indices, groups_r = makeDatasetDynamicPerlin(
        node_amt=node_amt,
        group_amt=group_amt,
        time_steps=time_steps,
        distance_threshold=distance_threshold,
        noise_scale=noise_scale,
        noise_strength=noise_strength,
        tilt_strength=tilt_strength,
        boundary=boundary,
        perlin_offset=perlin_offset_amt,
        mixed=False
        
    )

    # ego_index, ego_positions, ego_adjacency, ego_edge_indices, EgoMask = getEgo(all_positions_cpu, adjacency_dynamic_cpu, hop=2, union=False)
    ego_index, pruned_adj, reachable = getEgo(all_positions_cpu, adjacency_dynamic_cpu, hop=hops, union=False, min_groups=groups_r)

    plot_faster(all_positions_cpu, adjacency_dynamic_cpu, ego_idx=ego_index, ego_mask=reachable)

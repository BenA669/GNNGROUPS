import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from noise import pnoise2
from animate import plot_faster
import time as time


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
    intra_prob=0.05,
    inter_prob=0.001,
    noise_scale=1.0,      # scales the noise coordinate (affects frequency)
    noise_strength=0.1,   # multiplier for the noise gradient (step size)
    tilt_strength=0.05,   # added constant bias (tilt) for each group
    octaves=1,
    persistence=0.5,
    lacunarity=2.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(torch.initial_seed())

    # --- 1. Assign nodes to groups and set initial positions ---
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

    # --- 3. Update node positions over time based on noise gradient, tilt, and group speed ---
    for t in range(1, time_steps):
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

            # Combine noise gradient with the constant tilt.
            step_x = noise_strength * grad_x + tilt_strength * tilt_vector[0]
            step_y = noise_strength * grad_y + tilt_strength * tilt_vector[1]

            # Multiply the step by the group's speed multiplier.
            step_x *= speed_multiplier
            step_y *= speed_multiplier

            new_positions[i, 0] = prev_positions[i, 0] + step_x
            new_positions[i, 1] = prev_positions[i, 1] + step_y
            # The group id (stored in column 2) remains unchanged.
        all_positions[t] = new_positions

    # --- 4. Create a static (probabilistic) adjacency matrix ---
    node_groups = nodes[:, 2].long()  # shape: (node_amt,)
    same_group_mask = node_groups.unsqueeze(0).eq(node_groups.unsqueeze(1))  # (node_amt, node_amt)
    p_mat = torch.full((node_amt, node_amt), inter_prob, device=device)
    p_mat[same_group_mask] = intra_prob  # higher connection probability for same-group pairs

    rand_mat = torch.rand((node_amt, node_amt), device=device, generator=rng)
    static_adj_matrix = (rand_mat < p_mat).to(torch.int8)
    # Make adjacency symmetric and remove self-connections.
    static_adj_matrix = torch.triu(static_adj_matrix, diagonal=1)
    static_adj_matrix = static_adj_matrix + static_adj_matrix.T

    # --- 5. For each time step, add dynamic (distance–based) connections ---
    adjacency_dynamic = []
    for t in range(time_steps):
        positions_t = all_positions[t, :, :2]  # shape: (node_amt, 2)
        # Compute pairwise distances.
        dist_mat = torch.cdist(positions_t.unsqueeze(0), positions_t.unsqueeze(0)).squeeze(0)
        # Build dynamic adjacency from distance threshold.
        dist_adj = (dist_mat < distance_threshold).to(torch.int8)
        dist_adj.fill_diagonal_(0)
        dist_adj = torch.triu(dist_adj, diagonal=1) + torch.triu(dist_adj, diagonal=1).T
        # Combine static and dynamic connections (logical OR).
        union_adj = (static_adj_matrix | dist_adj).to(torch.int8)
        adjacency_dynamic.append(union_adj)

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

    return all_positions_cpu, adjacency_dynamic_cpu, edge_indices

def getEgo(all_positions_cpu, adjacency_dynamic_cpu):
    """
    Select a random ego node and extract its ego network from the dynamic dataset.
    
    The ego network consists of the chosen node and all nodes that have been connected
    to it (in any time step) according to the dynamic adjacency matrices.
    
    Parameters:
        all_positions_cpu (torch.Tensor): Tensor of shape (time_steps, node_amt, 3)
            containing the positions (and group id in column 2) for each node at each time step.
        adjacency_dynamic_cpu (torch.Tensor): Tensor of shape (time_steps, node_amt, node_amt)
            containing the dynamic (time-varying) adjacency matrices.
    
    Returns:
        ego_positions (torch.Tensor): Filtered positions tensor containing only the ego and its neighbors,
            of shape (time_steps, n_ego_nodes, 3).
        ego_adjacency (torch.Tensor): Filtered dynamic adjacency tensor of shape (time_steps, n_ego_nodes, n_ego_nodes).
    """
    # Get the total number of nodes and time steps.
    time_steps, total_nodes, _ = all_positions_cpu.shape

    # Choose a random ego node index.
    ego_idx = random.randint(0, total_nodes - 1)

    # Determine the ego's neighbors:
    # For each time step, check which nodes are connected to the ego node.
    # Then take the union (logical OR) over all time steps.
    # Note: Since the dynamic adjacency matrices are symmetric, you can use either
    # the row or the column corresponding to the ego.
    EgoMask = (adjacency_dynamic_cpu[:, ego_idx, :] > 0)
    EgoMask[:, ego_idx] = True
    union_neighbors = EgoMask.any(dim=0)
    union_neighbors[ego_idx] = True


    # Get the indices of the ego node and its neighbors.
    neighbor_indices = torch.nonzero(union_neighbors, as_tuple=False).flatten()

    
    # Filter the positions tensor to include only the selected nodes.
    ego_positions = all_positions_cpu[:, neighbor_indices, :]


    # For the dynamic adjacency, extract the submatrix (for each time step)
    # corresponding to the chosen nodes.
    ego_adjacency = adjacency_dynamic_cpu[:, neighbor_indices][:, :, neighbor_indices]

    ego_edge_indices = []
    for t in range(time_steps):
        adj_t = ego_adjacency[t]
        ego_edge_index_t = adjacency_to_edge_index(adj_t)
        ego_edge_indices.append(ego_edge_index_t)

    return ego_idx, ego_positions, ego_adjacency, ego_edge_indices, EgoMask
    


if __name__ == '__main__':
    

    time_steps = 20
    group_amt = 4
    node_amt = 400

    # Adjust these parameters as desired.
    noise_scale = 0.05      # frequency of the noise
    noise_strength = 2      # influence of the noise gradient
    tilt_strength = 0.25     # constant bias per group

    all_positions_cpu, adjacency_dynamic_cpu, edge_indices = makeDatasetDynamicPerlin(
        node_amt=node_amt,
        group_amt=group_amt,
        std_dev=1,
        time_steps=time_steps,
        distance_threshold=2,
        intra_prob=0.05,
        inter_prob=0.001,
        noise_scale=noise_scale,
        noise_strength=noise_strength,
        tilt_strength=tilt_strength,
        octaves=1,
        persistence=0.5,
        lacunarity=2.0
    )

    # Visualize the evolving graph using the animate module.
    # plot_faster(all_positions_cpu, adjacency_dynamic_cpu)

    ego_index, ego_positions, ego_adjacency, ego_edge_indices, EgoMask = getEgo(all_positions_cpu, adjacency_dynamic_cpu)

    # print(ego_adjacency.shape)
    print("GUH")
    
    plot_faster(all_positions_cpu, adjacency_dynamic_cpu, ego_idx=ego_index, ego_network_indices=ego_positions)

# makeDataset.py

import torch
import math

def makeDatasetDynamic(
        node_amt=400,
        group_amt=4,
        std_dev=1,
        speed_min=0.01,
        speed_max=0.5,
        time_steps=20,
        intra_prob=0.05,
        inter_prob=0.001
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(torch.initial_seed())

    rand_vals = torch.rand(node_amt, generator=rng, device=device)
    group_edges = torch.arange(1, group_amt + 1, device=device) * (1.0 / group_amt)
    groups = torch.bucketize(rand_vals, group_edges)  # each in [1, group_amt]

    last_seen_seeds = torch.rand((group_amt, 2), device=device)  # shape: (4, 2)
    nodes = torch.zeros((node_amt, 3), device=device)            # shape: (node_amt, 3)
    nodes[:, 2] = groups - 1  # Adjust group indices to start from 0

    for i in range(node_amt):
        g = int(groups[i].item()) - 1
        point = torch.normal(mean=last_seen_seeds[g], std=std_dev, generator=rng)
        last_seen_seeds[g] = point  # shift the group's seed
        nodes[i, 0] = point[0]
        nodes[i, 1] = point[1]

    group_velocities = torch.zeros((group_amt, 2), device=device)
    for g in range(group_amt):
        dir_vec = torch.randn(2, generator=rng, device=device)
        dir_vec = dir_vec / torch.norm(dir_vec)
        speed = torch.rand(1, generator=rng, device=device) * (speed_max - speed_min) + speed_min
        group_velocities[g] = speed * dir_vec

    all_positions = torch.zeros((time_steps, node_amt, 3), device=device)
    all_positions[0] = nodes

    for t in range(1, time_steps):
        prev_positions = all_positions[t - 1]
        new_positions = prev_positions.clone()

        # Get group for each node
        node_groups = prev_positions[:, 2].long()  # shape: (node_amt,)
        # Get the velocity for each node based on its group
        node_velocities = group_velocities[node_groups]  # shape: (node_amt, 2)
        # Update x and y positions
        new_positions[:, :2] += node_velocities

        all_positions[t] = new_positions

    node_groups = nodes[:, 2].long()  # shape: (node_amt,)

    # We can ignore adjacency here since adjacency is recomputed based on proximity
    adj_matrix = torch.zeros((node_amt, node_amt), dtype=torch.int8, device=device)

    all_positions_cpu = all_positions.cpu()
    adj_matrix_cpu = adj_matrix.cpu()

    return all_positions, adj_matrix, all_positions_cpu, adj_matrix_cpu

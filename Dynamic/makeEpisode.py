import random
import matplotlib.pyplot as plt
import numpy as np
import torch


def makeDatasetDynamic(
        node_amt=400,
        group_amt=4,
        std_dev=1,

        speed_min=0.01,
        speed_max=0.5,
        time_steps=20,

        intra_prob=0.05,
        inter_prob=0.001,

        distance_threshold=0.5  # distance under which nodes get connected
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(torch.initial_seed())

    # ---- 1) Assign each node to a group. ----
    rand_vals = torch.rand(node_amt, generator=rng, device=device)
    group_edges = torch.arange(1, group_amt + 1, device=device) * (1.0 / group_amt)
    groups = torch.bucketize(rand_vals, group_edges)  # each in [0, group_amt - 1]

    # ---- 2) Generate initial positions for each group. ----
    last_seen_seeds = torch.rand((group_amt, 2), device=device)  # shape: (group_amt, 2)
    nodes = torch.zeros((node_amt, 3), device=device)             # shape: (node_amt, 3)
    nodes[:, 2] = groups

    for i in range(node_amt):
        g = int(groups[i].item())
        point = torch.normal(mean=last_seen_seeds[g], std=std_dev, generator=rng)
        # Shift the group's "seed" to get a random walk for where the group center might go
        last_seen_seeds[g] = point
        nodes[i, 0] = point[0]
        nodes[i, 1] = point[1]

    # ---- 3) Assign velocities to each group. ----
    group_velocities = torch.zeros((group_amt, 2), device=device)
    for g in range(group_amt):
        dir_vec = torch.randn(2, generator=rng, device=device)
        dir_vec = dir_vec / torch.norm(dir_vec)  # random unit direction
        speed = torch.rand(1, generator=rng, device=device) * (speed_max - speed_min) + speed_min
        group_velocities[g] = speed * dir_vec

    # ---- 4) Compute all positions for each time-step. ----
    all_positions = torch.zeros((time_steps, node_amt, 3), device=device)
    all_positions[0] = nodes

    for t in range(1, time_steps):
        prev_positions = all_positions[t - 1]
        new_positions = prev_positions.clone()

        node_groups = prev_positions[:, 2].long()        # shape: (node_amt,)
        node_velocities = group_velocities[node_groups]  # shape: (node_amt, 2)
        new_positions[:, :2] += node_velocities

        all_positions[t] = new_positions

    # ---- 5) Create the *static* adjacency due to random inter/intra-group probabilities. ----
    #         This adjacency does NOT change over time.
    node_groups = nodes[:, 2].long()  # shape: (node_amt,)
    same_group_mask = node_groups.unsqueeze(0).eq(node_groups.unsqueeze(1))  # (node_amt, node_amt)

    # Create a matrix of probabilities
    p_mat = torch.full((node_amt, node_amt), inter_prob, device=device)
    p_mat[same_group_mask] = intra_prob  # fill in same-group entries

    # Sample from uniform(0,1) in an NxN matrix
    rand_mat = torch.rand((node_amt, node_amt), device=device, generator=rng)

    # Build adjacency by thresholding
    static_adj_matrix = (rand_mat < p_mat).to(torch.int8)

    # Make adjacency symmetric and remove diagonal
    static_adj_matrix = torch.triu(static_adj_matrix, diagonal=1)
    static_adj_matrix = static_adj_matrix + static_adj_matrix.T  # symmetrize
    # diagonal is automatically 0 from triu(..., diagonal=1)

    # ---- 6) For each time-step, compute a distance-based adjacency, then union it with the static one. ----
    # adjacency_dynamic will be shape: (time_steps, node_amt, node_amt)
    adjacency_dynamic = []
    for t in range(time_steps):
        positions_t = all_positions[t, :, :2]  # shape: (node_amt, 2)

        # Compute pairwise distances
        # (You can do this more efficiently with broadcasting or cdist, but we'll do it explicitly here.)
        dist_mat = torch.cdist(positions_t.unsqueeze(0), positions_t.unsqueeze(0)).squeeze(0)
        # dist_mat is shape (node_amt, node_amt)

        # Build adjacency by threshold
        dist_adj = (dist_mat < distance_threshold).to(torch.int8)
        # Remove diagonal
        dist_adj.fill_diagonal_(0)
        # Make sure it's symmetric (though cdist should already yield a symmetric matrix)
        # We'll just force-symmetrize it in case of numerical issues:
        dist_adj = torch.triu(dist_adj, diagonal=1) + torch.triu(dist_adj, diagonal=1).T

        # Combine with static adjacency
        union_adj = (static_adj_matrix | dist_adj).to(torch.int8)

        adjacency_dynamic.append(union_adj)

    # Convert list -> tensor: (time_steps, node_amt, node_amt)
    adjacency_dynamic = torch.stack(adjacency_dynamic, dim=0)

    # Move to CPU for convenience
    all_positions_cpu = all_positions.cpu()
    adjacency_dynamic_cpu = adjacency_dynamic.cpu()

    return all_positions_cpu, adjacency_dynamic_cpu


if __name__ == '__main__':
    time_steps = 20
    group_amt = 4
    node_amt = 400

    all_positions_cpu, adjacency_dynamic_cpu = makeDatasetDynamic(
        node_amt=node_amt,
        group_amt=group_amt,
        time_steps=time_steps,
        distance_threshold=2,  # Adjust as needed
        intra_prob=0.05,
        inter_prob=0.001
    )

    # --- Visualization ---
    colors = ["blue", "orange", "green", "red", "purple", "brown"]  # extend as needed

    plt.figure()
    for t in range(time_steps):
        plt.clf()  # Clear the current figure for the new time-step

        # Plot edges first
        adj_t = adjacency_dynamic_cpu[t]  # shape: (node_amt, node_amt)
        for i in range(node_amt):
            # only need to iterate j in (i+1 .. node_amt) to avoid double-plotting
            for j in range(i + 1, node_amt):
                if adj_t[i, j] == 1:
                    x_vals = [all_positions_cpu[t, i, 0].item(), all_positions_cpu[t, j, 0].item()]
                    y_vals = [all_positions_cpu[t, i, 1].item(), all_positions_cpu[t, j, 1].item()]
                    plt.plot(x_vals, y_vals, c="gray", alpha=0.3, linewidth=0.5)

        # Plot the nodes in scatter form, color-coded by their group
        group_ids = all_positions_cpu[t, :, 2].long()
        unique_groups = torch.unique(group_ids)
        for g in unique_groups:
            mask = (group_ids == g)
            plt.scatter(
                all_positions_cpu[t, mask, 0],
                all_positions_cpu[t, mask, 1],
                c=colors[g % len(colors)],
                label=f"Group {g}",
                alpha=0.7,
            )

        plt.title(f"Time Step {t}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.pause(0.01)  # pause briefly so plots update

    plt.show()

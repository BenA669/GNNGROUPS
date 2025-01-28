import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.animation as animation
import torch

def animate_positions(all_positions, labels, interval=200):
    """
    Creates and displays an animation of node positions over time.
    all_positions: shape (timeSteps, nodeAmount, 2)
    labels: shape (nodeAmount,)
    """
    # Move data to CPU once
    all_positions_cpu = all_positions.cpu().numpy()  # (timeSteps, nodeAmount, 2)
    labels_cpu = labels.cpu().numpy()                # (nodeAmount,)

    timeSteps, nodeAmount, _ = all_positions_cpu.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)  # Adjust if necessary
    ax.set_ylim(0, 1)

    # --- Create the scatter plot using the 0th frame ---
    scatter = ax.scatter(
        all_positions_cpu[0, :, 0],
        all_positions_cpu[0, :, 1],
        c=labels_cpu,            # color by group
        cmap="tab10",            # choose a color map
        s=40,
        vmin=0, vmax=labels_cpu.max()
    )

    # Optional: colorbar
    # plt.colorbar(scatter, ax=ax, label="Group")

    # initialization function for FuncAnimation
    def init():
        # Set the offsets to the positions at t=0
        scatter.set_offsets(all_positions_cpu[0])
        ax.set_title(f"Time step: 0")
        return (scatter,)

    # update function
    def update(frame):
        scatter.set_offsets(all_positions_cpu[frame])
        ax.set_title(f"Time step: {frame}")
        return (scatter,)

    ani = animation.FuncAnimation(
        fig,                    # figure object
        update,                 # update function
        frames=range(timeSteps),
        init_func=init,         # initialization
        interval=interval,      # delay between frames in ms
        blit=True
    )

    plt.show()


def makeDynamicDataSetCUDA(groupsAmount=2,
                           nodeAmount=100,
                           nodeDim=2,
                           timeSteps=10,
                           nodeNeighborStdDev=0.2,
                           intra_group_prob=0.09,
                           inter_group_prob=0.005,
                           speed_min=0.01,
                           speed_max=0.05,
                           recalc_adjacency=False):
    """
    Generates a dynamic dataset where each group moves at a constant velocity.
    
    Args:
        groupsAmount (int):   Number of groups
        nodeAmount (int):     Total number of nodes
        nodeDim (int):        Dimensionality of node positions (default 2)
        timeSteps (int):      Number of time steps in the dynamic sequence
        nodeNeighborStdDev (float): Standard deviation for node clustering around group seeds
        intra_group_prob (float): Probability factor for edges within a group
        inter_group_prob (float): Probability factor for edges between groups
        speed_min (float):    Minimum speed for group velocity
        speed_max (float):    Maximum speed for group velocity
        recalc_adjacency (bool): Whether to recalc adjacency at each time step
        
    Returns:
        all_positions (torch.Tensor): Positions for all time steps; shape = (timeSteps, nodeAmount, nodeDim).
        adj_matrices (List[torch.Tensor] or torch.Tensor): If `recalc_adjacency=True`, this will be a list
            of adjacency matrices for each time step. Otherwise, it will be just the initial adjacency matrix.
        labels (torch.Tensor): Labels of shape (nodeAmount,).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device).manual_seed(torch.seed())

    if nodeAmount % groupsAmount != 0:
        raise ValueError("Node amount must be divisible by groups amount")

    node_per_group = nodeAmount // groupsAmount

    # ~~~ Step 1: Generate the static data for t=0 using your existing approach ~~~
    # data shape = (groupsAmount, node_per_group, nodeDim)
    data = torch.zeros((groupsAmount, node_per_group, nodeDim), 
                       dtype=torch.float32, device=device)

    # 1a) Initialize the "seed" for each group (just as an example, can be random or structured)
    for group in range(groupsAmount):
        data[group, 0] = torch.rand(nodeDim, generator=rng, device=device)
    
    # 1b) Populate nodes around each group's seed
    for group in range(groupsAmount):
        group_avg = data[group, 0].clone()
        for node in range(1, node_per_group):
            point = None
            while True:
                # Sample position around current group average
                candidate_point = torch.normal(mean=group_avg, 
                                               std=nodeNeighborStdDev, 
                                               generator=rng)
                # Force positions to remain in [0,1] x [0,1]
                if (candidate_point >= 0).all() and (candidate_point <= 1).all():
                    point = candidate_point
                    break
            data[group, node] = point

    # Flatten data to shape (nodeAmount, nodeDim)
    all_nodes_t0 = data.view(nodeAmount, nodeDim)

    # 1c) Create labels
    labels = torch.arange(groupsAmount, device=device).repeat_interleave(node_per_group)

    # 1d) Shuffle nodes and labels
    shuffle_indices = torch.randperm(nodeAmount, generator=rng, device=device)
    shuffled_nodes_t0 = all_nodes_t0[shuffle_indices]
    labels = labels[shuffle_indices]

    # ~~~ Step 2: Build adjacency for t=0 (optional to recalc every time step) ~~~
    # Because adjacency can be large, store it as int or bool
    def compute_adjacency(positions):
        # positions: shape = (nodeAmount, nodeDim)
        adjacency_matrix = torch.zeros((nodeAmount, nodeAmount), dtype=torch.int32, device=device)

        # compute pairwise distances
        distances = torch.cdist(positions, positions, p=2)

        # group membership
        group_indices = torch.arange(nodeAmount, device=device) // node_per_group
        # But note: we shuffled, so we want the group of the *shuffled* index
        # Actually, let's do it the simpler way: we already have labels, which is the group id:
        # labels[i] gives group for node i in the shuffled arrangement
        # So:
        label_expand1 = labels.view(-1, 1)
        label_expand2 = labels.view(1, -1)
        intra_mask = (label_expand1 == label_expand2)

        # Compute connection probabilities
        # distances + 1 to avoid division by zero
        probabilities = torch.where(intra_mask, 
                                    intra_group_prob / (1 + distances), 
                                    inter_group_prob / (1 + distances))

        # Random sample to determine edges
        random_probs = torch.rand((nodeAmount, nodeAmount), generator=rng, device=device)
        adjacency_matrix = (random_probs < probabilities).int()

        # Symmetrize adjacency
        adjacency_matrix = torch.triu(adjacency_matrix, diagonal=1)
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        return adjacency_matrix

    adj_t0 = compute_adjacency(shuffled_nodes_t0)

    # ~~~ Step 3: Assign a random velocity to each group ~~~
    # shape = (groupsAmount, nodeDim); each group has its own velocity
    # Speed is random in [speed_min, speed_max], direction is random
    group_velocities = torch.zeros(groupsAmount, nodeDim, device=device)
    for group in range(groupsAmount):
        # Random direction
        dir_vec = torch.randn(nodeDim, generator=rng, device=device)
        dir_vec = dir_vec / torch.norm(dir_vec)  # normalize
        # Random speed
        speed = torch.rand(1, generator=rng, device=device) * (speed_max - speed_min) + speed_min
        # velocity = speed * direction
        group_velocities[group] = speed * dir_vec

    # ~~~ Step 4: Evolve positions over time ~~~
    # We'll store positions for each time step in a (timeSteps, nodeAmount, nodeDim) tensor
    all_positions = torch.zeros((timeSteps, nodeAmount, nodeDim), device=device)
    # Set time t=0
    all_positions[0] = shuffled_nodes_t0

    # If recomputing adjacency at each step, store each adjacency in a list
    adj_matrices = []

    if recalc_adjacency:
        # adjacency for t=0
        adj_matrices.append(adj_t0)
    else:
        # just store the initial adjacency
        adj_matrices = adj_t0

    # We must remember which group each node belongs to *after shuffle*
    # because the label array is aligned to the shuffled arrangement
    # label i => group
    # so group_velocities[ labels[i] ] is the velocity of node i
    for t in range(1, timeSteps):
        prev_positions = all_positions[t-1]

        # Update each node based on which group it belongs to
        new_positions = prev_positions.clone()
        for i in range(nodeAmount):
            grp = labels[i]
            new_positions[i] = prev_positions[i] + group_velocities[grp]

        # (Optional) keep bounding in [0, 1]?  E.g. bounce or wrap
        # Here’s a simple "wrap around" example:
        new_positions = new_positions % 1.0

        all_positions[t] = new_positions

        if recalc_adjacency:
            adj_t = compute_adjacency(new_positions)
            adj_matrices.append(adj_t)

    return all_positions, adj_matrices, labels


# ~~~ EXAMPLE USAGE ~~~
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groupsAmount = 2
    nodeAmount = 50
    timeSteps = 20
    intra_group_prob = 0.4
    inter_group_prob = 0.01
    nodeNeighborStdDev = 0.05

    all_positions, adj_matrices, labels = makeDynamicDataSetCUDA(
        groupsAmount=groupsAmount,
        nodeAmount=nodeAmount,
        nodeNeighborStdDev=nodeNeighborStdDev,
        intra_group_prob=intra_group_prob,
        inter_group_prob=inter_group_prob,
        timeSteps=timeSteps,
        recalc_adjacency=False  # set True if you want adjacency updated at each time step
    )

    print("Shape of all_positions:", all_positions.shape)
    # -> (timeSteps, nodeAmount, nodeDim)
    if isinstance(adj_matrices, list):
        print("Adjacency matrices over time:", len(adj_matrices), adj_matrices[0].shape)
    else:
        print("Single adjacency matrix shape:", adj_matrices.shape)

    print("Labels shape:", labels.shape)

    animate_positions(all_positions, labels, interval=300)

    # # Quick demonstration of how you might visualize a few time steps
    # # (adapt or wrap your existing `plot_dataset` to handle multiple timesteps).
    # import matplotlib.pyplot as plt
    
    # # for t in [0, timeSteps//2, timeSteps-1]:
    # for t in range(timeSteps-1):
    #     fig, ax = plt.subplots(figsize=(6,6))
    #     positions_cpu = all_positions[t].cpu().numpy()
    #     ax.set_title(f"Node positions at time t={t}")

    #     # color by label
    #     unique_groups = torch.unique(labels)
    #     for g in unique_groups:
    #         group_indices = (labels == g).nonzero(as_tuple=True)[0]
    #         # Convert the index to CPU + NumPy:
    #         group_indices_cpu = group_indices.cpu().numpy()
            
    #         ax.scatter(positions_cpu[group_indices_cpu, 0],
    #                 positions_cpu[group_indices_cpu, 1],
    #                 label=f"Group {g.item()}")

    #     # ax.set_xlim([0,1])
    #     # ax.set_ylim([0,1])
    #     ax.legend()
    #     plt.show()

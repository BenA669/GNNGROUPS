import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

start = time.time()
# Determine the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = torch.Generator(device).manual_seed(torch.seed())

node_amt = 400
group_amt = 4
std_dev = 1

speed_min=0.01
speed_max=0.5
time_steps = 20

intra_prob = 0.01
inter_prob = 0.0001

# ------------------------------------------------------------------------------
# Setup node positions
# ------------------------------------------------------------------------------
nodes = torch.zeros((node_amt, 3), device=device)

# Assign each node to a group
for node in range(node_amt):
    p_G = torch.rand(1, generator=rng, device=device).item()
    for group in range(group_amt):
        if p_G <= (group + 1) * (1/group_amt):
            nodes[node, 2] = group
            break

# Generate positions by "diffusing" from group-specific seeds
last_seen_seeds = torch.rand((group_amt, 2), device=device)
for node in range(node_amt):
    node_group = int(nodes[node, 2].item())
    point = torch.normal(mean=last_seen_seeds[node_group], generator=rng, std=std_dev)
    last_seen_seeds[node_group] = point
    nodes[node, 0] = point[0]
    nodes[node, 1] = point[1] 

# ------------------------------------------------------------------------------
# Generate group velocities
# ------------------------------------------------------------------------------
group_velocities = torch.zeros((group_amt, 2), device=device)
for group in range(group_amt):
    dir_vec = torch.randn(2, generator=rng, device=device)
    dir_vec = dir_vec / torch.norm(dir_vec)
    speed = torch.rand(1, generator=rng, device=device) * (speed_max - speed_min) + speed_min
    group_velocities[group] = speed * dir_vec

# ------------------------------------------------------------------------------
# Prepare a tensor to store positions over time
# ------------------------------------------------------------------------------
all_positions = torch.zeros((time_steps, node_amt, 3), device=device)
all_positions[0] = nodes

# ------------------------------------------------------------------------------
# Update positions over time
# ------------------------------------------------------------------------------
for t in range(1, time_steps):
    prev_positions = all_positions[t-1]
    new_positions = prev_positions.clone()

    for node in range(node_amt):
        group = int(prev_positions[node, 2].item())
        velocity = group_velocities[group]
        new_positions[node, :2] += velocity  # Update x and y positions

    all_positions[t] = new_positions

# Move everything to CPU for edge creation and plotting
all_positions = all_positions.cpu()
nodes = nodes.cpu()

# ------------------------------------------------------------------------------
# Create adjacency matrix (Tensor)
#   - same group -> 10% chance of edge
#   - different group -> 1% chance of edge
# ------------------------------------------------------------------------------
adj_matrix = torch.zeros((node_amt, node_amt), dtype=torch.int8)  # or bool/int64

for i in range(node_amt):
    for j in range(i + 1, node_amt):
        group_i = int(nodes[i, 2].item())
        group_j = int(nodes[j, 2].item())
        if group_i == group_j:
            p = intra_prob  # same group
        else:
            p = inter_prob  # different groups

        # Sample probability
        if random.random() < p:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

end = time.time()

print("Time: ")
print(end - start)
# --- Visualization ---
colors = ["blue", "orange", "green", "red", "purple", "brown"]  # extend as needed

plt.figure()
for t in range(time_steps):
    plt.clf()  # Clear the current figure for the new time-step
    
    # Plot edges first
    # We'll scan through the upper triangular part of adj_matrix to draw edges.
    # (You can also use 'nonzero()' on a triangular portion to optimize.)
    for i in range(node_amt):
        for j in range(i + 1, node_amt):
            if adj_matrix[i, j] == 1:
                # Positions at time t
                x_vals = [all_positions[t, i, 0].item(), all_positions[t, j, 0].item()]
                y_vals = [all_positions[t, i, 1].item(), all_positions[t, j, 1].item()]
                plt.plot(x_vals, y_vals, c="gray", alpha=0.5, linewidth=0.5)

    # Scatter plot of nodes (by group)
    for g in range(group_amt):
        mask = (all_positions[t, :, 2] == g)
        plt.scatter(
            all_positions[t, mask, 0],
            all_positions[t, mask, 1],
            c=colors[g % len(colors)],
            label=f"Group {g}",
            alpha=0.7,
        )
    
    plt.title(f"Time Step {t}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.pause(0.0001)  # Adjust as needed

plt.show()

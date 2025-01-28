import random
import matplotlib.pyplot as plt
import numpy as np
import torch

# Determine the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = torch.Generator(device).manual_seed(torch.seed())


fig = plt.figure()

plt.title("New Dataset")

node_amt = 100
group_amt = 2
std_dev = 1

nodes = torch.zeros((3, node_amt), device=device)

print(nodes.shape)

node_per_group = node_amt//group_amt

for group in range(group_amt):
    seed = torch.rand(2, generator=rng, device=device)
    for node in range(node_per_group):
        point = torch.normal(mean=seed, generator=rng, std=std_dev,)

        node_index = node+(group*node_per_group)

        nodes[0, node_index] = point[0]
        nodes[1, node_index] = point[1] 
        nodes[2, node_index] = group

        seed = point

# Detach nodes from GPU if necessary
nodes = nodes.cpu()

# Plot the points with colors based on group labels
for group in range(group_amt):
    group_indices = nodes[2, :] == group
    plt.scatter(
        nodes[0, group_indices],  # x-coordinates
        nodes[1, group_indices],  # y-coordinates
        label=f"Group {group}"
    )

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Visualized Dataset")
plt.show()
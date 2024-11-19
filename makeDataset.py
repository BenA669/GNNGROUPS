import numpy as np
import matplotlib.pyplot as plt
import torch

def makePlotWithErrors(data, num_groups, adjacency_matrix, node_positions, true_labels, predicted_labels):
    colors = plt.cm.get_cmap('tab10', num_groups)
    plt.figure(figsize=(7, 7))

    # Draw edges first based on adjacency matrix
    node_count = len(node_positions)
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if adjacency_matrix[i, j] == 1:
                plt.plot(
                    [node_positions[i, 0], node_positions[j, 0]],
                    [node_positions[i, 1], node_positions[j, 1]],
                    'gray', linewidth=0.5, alpha=0.3
                )

    for group_idx in range(num_groups):
        group_points = data[group_idx]
        group_true_labels = true_labels[group_idx * len(group_points):(group_idx + 1) * len(group_points)]
        group_predicted_labels = predicted_labels[group_idx * len(group_points):(group_idx + 1) * len(group_points)]

        # Plot all nodes except the seed node
        for i, (point, true_label, predicted_label) in enumerate(zip(group_points[1:], group_true_labels[1:], group_predicted_labels[1:])):
            if true_label == predicted_label:
                plt.scatter(point[0], point[1], color=colors(group_idx), label=f'Group {group_idx + 1}' if i == 0 else "")
            else:
                plt.scatter(point[0], point[1], color='red', label='Incorrect' if i == 0 else "")

        # Plot the seed node with a different color or marker
        plt.scatter(group_points[0, 0], group_points[0, 1], color='black', label=f'Seed {group_idx + 1}')

    # Adding labels, legend, and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Visualization of Points from {num_groups} Groups with Errors Highlighted')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
    
def makePlot(data, num_groups, adjacency_matrix, node_positions):
    colors = plt.cm.get_cmap('tab10', num_groups)
    plt.figure(figsize=(7, 7))

     # Draw edges first based on adjacency matrix
    node_count = len(node_positions)
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if adjacency_matrix[i, j] == 1:
                plt.plot(
                    [node_positions[i, 0], node_positions[j, 0]],
                    [node_positions[i, 1], node_positions[j, 1]],
                    'gray', linewidth=0.5, alpha=0.3
                )

    for group_idx in range(num_groups):
        group_points = data[group_idx]
        # Plot all nodes except the seed node
        plt.scatter(group_points[1:, 0], group_points[1:, 1], color=colors(group_idx), label=f'Group {group_idx + 1}')
        # Plot the seed node with a different color or marker
        plt.scatter(group_points[0, 0], group_points[0, 1], color='black', label=f'Seed {group_idx + 1}')

    # Adding labels, legend, and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Visualization of Points from {num_groups} Groups')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def makeDataSet(groupsAmount=2, nodeAmount=100, nodeDim=2, nodeNeighborBaseProb=0.95, nodeNeighborStdDev=0.1, connectedThreshold=0.05, intra_group_prob=0.08, inter_group_prob=0.005, repulsion_factor=0.2):
    rng = np.random.default_rng()  # Generates different Seed every time
    nodePerGroup = int(nodeAmount / groupsAmount)

    data = np.zeros(shape=(groupsAmount, nodePerGroup, nodeDim), dtype=float)
    outliers = list()
    allGroupAverages = list()
    for group in range(groupsAmount):
        if group == 0:
            data[group, 0] = rng.random(nodeDim)
        else:
            # Attempt to place the seed node further from other groups
            while True:
                candidate_position = rng.random(nodeDim)
                min_distance = min(
                    np.linalg.norm(candidate_position - data[other_group, 0])
                    for other_group in range(group)
                )
                if min_distance > repulsion_factor:
                    data[group, 0] = candidate_position
                    break

        groupAverage = data[group, 0]
        for node in range(1, nodePerGroup):
            # Get running Average of group every 10%:
            if node % (nodePerGroup / 10) == 0:
                for dim in range(nodeDim):
                    if (len(data[group][0:node-1]) > 1):
                        # print(data[group][0:node-1][:, dim])
                        groupAverage[dim] = np.mean(data[group][0:node-1][:, dim])
                # print("Running GA: {}".format(groupAverage))
                
            p = rng.random() # 0, 1
            # p = 0
            if (p < nodeNeighborBaseProb):            
                for dim in range(nodeDim):
                    data[group, node, dim] = rng.normal(loc=groupAverage[dim], scale=nodeNeighborStdDev)
            else:
                #Seperate Calc
                # Mark outlier
                outliers.append(node)
                for dim in range(nodeDim):
                    data[group, node, dim] = rng.normal(loc=groupAverage[dim], scale=nodeNeighborStdDev)

        allGroupAverages.append(groupAverage)
        # Make outliers
    # print(allGroupAverages)
    average_array = np.mean(allGroupAverages, axis=0)
    for group in range(groupsAmount):
        for node in outliers:
            # data[group, node] = rng.random(nodeDim)
            data[group, node] = rng.normal(loc=average_array, scale=nodeNeighborStdDev)
        # print("Outlier Amt: {}".format(len(outliers)/nodeAmount))
        # print("Final group average for group {} is {}".format(group, groupAverage))

    # Combine all nodes into one array
    all_nodes = data.reshape(nodeAmount, nodeDim)

    # Create adjacency matrix based on probabilities
    adjacency_matrix = np.zeros((nodeAmount, nodeAmount), dtype=int)

    # Build adjacency matrix based on both distance threshold and group probabilities
    for i in range(nodeAmount):
        for j in range(i + 1, nodeAmount):
            group_i = i // nodePerGroup
            group_j = j // nodePerGroup

            # Calculate the Euclidean distance between nodes i and j
            distance = np.linalg.norm(all_nodes[i] - all_nodes[j])

            if distance <= connectedThreshold:
                # Assign higher probability if nodes are in the same group, otherwise lower probability
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Symmetric matrix
                continue

            if group_i == group_j:
                prob = intra_group_prob / (1 + distance)  # Decrease probability with distance
            else:
                prob = inter_group_prob / (1 + distance)  # Decrease probability with distance

            # Determine if the nodes should be connected
            if rng.random() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Symmetric matrix
    # makePlot(data, groupsAmount, adjacency_matrix, all_nodes)

    labels = np.array([i // (nodeAmount // groupsAmount) for i in range(nodeAmount)])

    all_nodes = torch.from_numpy(all_nodes)
    labels = torch.from_numpy(labels).long()
    adjacency_matrix = torch.from_numpy(adjacency_matrix)

    shuffle_indices = torch.randperm(all_nodes.size(0))
    shuffled_all_nodes = all_nodes[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]

    shuffled_adj = adjacency_matrix[shuffle_indices][:, shuffle_indices]
    
    # print(adjacency_matrix)
    # print(shuffled_adj)
    return torch.from_numpy(data), shuffled_adj, shuffled_all_nodes, shuffled_labels
    # return torch.from_numpy(data), adjacency_matrix, all_nodes, labels

    


if __name__ == "__main__":
    # data, adj, all_nodes, labels = makeDataSet(groupsAmount=2, intra_group_prob=0.1, inter_group_prob=0.01)
    data, adj, all_nodes, labels = makeDataSet(nodeAmount=100)

    makePlot(data, 2, adj, all_nodes)
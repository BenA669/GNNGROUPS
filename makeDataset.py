import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx

# Convert tensors to numpy for plotting
def plot_dataset(data, num_groups, adjacency_matrix, node_positions, true_labels, predicted_labels=None):
    data = data.numpy()
    adjacency_matrix = adjacency_matrix.numpy()
    node_positions = node_positions.numpy()
    true_labels = true_labels.numpy()
    if predicted_labels is not None and type(predicted_labels) == torch.Tensor:
        predicted_labels = predicted_labels.numpy()

    # Create a NetworkX graph from the adjacency matrix
    G = nx.Graph()
    for i in range(len(node_positions)):
        G.add_node(i, pos=node_positions[i])

    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix)):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = {i: node_positions[i] for i in range(len(node_positions))}
    
    # Set up the plot
    plt.figure(figsize=(10, 10))
    plt.title("Scatter Plot with Incorrect Labels Highlighted")

    # Plot nodes based on true labels
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'cyan'][:num_groups]
    for group in range(num_groups):
        group_nodes = [i for i in range(len(true_labels)) if true_labels[i] == group]
        group_positions = node_positions[group_nodes]
        plt.scatter(group_positions[:, 0], group_positions[:, 1], label=f"Group {group}", alpha=0.8, color=colors[group])

    # Overlay incorrect labels in transparent red
    if (predicted_labels is not None):
        incorrect_nodes = [i for i in range(len(predicted_labels)) if predicted_labels[i] != true_labels[i]]
        if len(incorrect_nodes) > 0:
            incorrect_positions = node_positions[incorrect_nodes]
            plt.scatter(incorrect_positions[:, 0], incorrect_positions[:, 1], color='red', alpha=0.2, s=100, label='Incorrect Predictions')

    # Plot the edges
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')

    # Plot settings
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()

def makeDataSet(groupsAmount=2, nodeAmount=100, nodeDim=2, nodeNeighborBaseProb=0.9, nodeNeighborStdDev=0.085, connectedThreshold=0.05, intra_group_prob=0.09, inter_group_prob=0.005, repulsion_factor=0.2):
    if nodeAmount % groupsAmount != 0:
        print("Node amount must be divisible by groups amount")
        return
    rng = np.random.default_rng()  # Generates different Seed every time
    nodePerGroup = int(nodeAmount // groupsAmount)

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
    average_array = np.mean(allGroupAverages, axis=0)
    for group in range(groupsAmount):
        for node in outliers:
            data[group, node] = rng.normal(loc=average_array, scale=nodeNeighborStdDev)

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

            

            if group_i == group_j:
                prob = intra_group_prob / (1 + distance)  # Decrease probability with distance
            else:
                prob = inter_group_prob / (1 + distance)  # Decrease probability with distance
            
            if distance <= connectedThreshold:
                # Assign higher probability if nodes are in the same group, otherwise lower probability
                prob *= 2

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
    groupsAmount = 3
    nodeAmount = 135
    # data, adj, all_nodes, labels = makeDataSet(groupsAmount=2, intra_group_prob=0.1, inter_group_prob=0.01)
    data, adj, all_nodes, labels = makeDataSet(nodeNeighborStdDev=0.2, nodeNeighborBaseProb = 1, nodeAmount=nodeAmount, groupsAmount=groupsAmount, repulsion_factor=0.5)

    plot_dataset(data, groupsAmount, adj, all_nodes, labels)
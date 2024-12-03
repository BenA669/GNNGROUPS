import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx
from tqdm import tqdm

# Convert tensors to numpy for plotting
def plot_dataset(data, adjacency_matrix, node_positions, true_labels, predicted_labels=None):
    data = data.cpu().numpy()
    adjacency_matrix = adjacency_matrix.cpu().numpy()
    node_positions = node_positions.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    if predicted_labels is not None and type(predicted_labels) == torch.Tensor:
        predicted_labels = predicted_labels.cpu().numpy()

    num_groups = len(np.unique(true_labels))

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
        plt.scatter(group_positions[:, 0], group_positions[:, 1], label=f"Group {group}", alpha=0.8, color=colors[group%8])

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

def makeDataSetOLD(groupsAmount=2, nodeAmount=100, nodeDim=2, nodeNeighborBaseProb=0.9, nodeNeighborStdDev=0.085, connectedThreshold=0.05, intra_group_prob=0.09, inter_group_prob=0.005, repulsion_factor=0.2):
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
    data = torch.from_numpy(data)

    shuffle_indices = torch.randperm(all_nodes.size(0))
    shuffled_all_nodes = all_nodes[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]

    shuffled_adj = adjacency_matrix[shuffle_indices][:, shuffle_indices]
    
    # print(adjacency_matrix)
    # print(shuffled_adj)
    return data, shuffled_adj, shuffled_all_nodes, shuffled_labels
    # return torch.from_numpy(data), adjacency_matrix, all_nodes, labels


def makeDataSet(groupsAmount=2, nodeAmount=100, nodeDim=2, nodeNeighborBaseProb=0.9, nodeNeighborStdDev=0.085, connectedThreshold=0.05, intra_group_prob=0.09, inter_group_prob=0.005, repulsion_factor=0.2):
    rng = torch.Generator().manual_seed(torch.seed())  # Generates different seed every time
    node_per_group = nodeAmount // groupsAmount

    data = torch.zeros((groupsAmount, node_per_group, nodeDim), dtype=torch.float32)
    outliers = []
    all_group_averages = []

    for group in range(groupsAmount):
        if group == 0:
            data[group, 0] = torch.rand(nodeDim, generator=rng)
        else:
            while True:
                candidate_position = torch.rand(nodeDim, generator=rng)
                min_distance = torch.min(torch.linalg.norm(candidate_position - data[:group, 0], dim=1))
                if min_distance > repulsion_factor:
                    data[group, 0] = candidate_position
                    break

        group_average = data[group, 0].clone()
        
        for node in range(1, node_per_group):
            # Running average update every 10% of nodes
            if node % (node_per_group // 10) == 0:
                group_average = torch.mean(data[group, :node], dim=0)

            p = torch.rand(1, generator=rng).item()

            if p < nodeNeighborBaseProb:
                data[group, node] = torch.normal(mean=group_average, std=nodeNeighborStdDev, generator=rng)
            else:
                outliers.append((group, node))
                data[group, node] = torch.normal(mean=group_average, std=nodeNeighborStdDev, generator=rng)

        all_group_averages.append(group_average)

    # Create outliers using global average
    average_array = torch.mean(torch.stack(all_group_averages), dim=0)
    for group, node in outliers:
        data[group, node] = torch.normal(mean=average_array, std=nodeNeighborStdDev, generator=rng)

    # Combine all nodes into one array
    all_nodes = data.view(nodeAmount, nodeDim)

    # Create adjacency matrix based on distances and group probabilities
    adjacency_matrix = torch.zeros((nodeAmount, nodeAmount), dtype=torch.int32)
    
    for i in range(nodeAmount):
        for j in range(i + 1, nodeAmount):
            group_i = i // node_per_group
            group_j = j // node_per_group

            # Calculate the Euclidean distance between nodes i and j
            distance = torch.linalg.norm(all_nodes[i] - all_nodes[j])

            if group_i == group_j:
                prob = intra_group_prob / (1 + distance)
            else:
                prob = inter_group_prob / (1 + distance)

            if distance <= connectedThreshold:
                prob *= 2

            if torch.rand(1, generator=rng).item() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Symmetric matrix

    # Create labels
    labels = torch.arange(groupsAmount).repeat_interleave(node_per_group)

    # Shuffle nodes and their labels
    shuffle_indices = torch.randperm(nodeAmount, generator=rng)
    shuffled_all_nodes = all_nodes[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]
    shuffled_adj = adjacency_matrix[shuffle_indices][:, shuffle_indices]

    return data, shuffled_adj, shuffled_all_nodes, shuffled_labels

def makeDataSetCUDA(groupsAmount=2, nodeAmount=100, nodeDim=2, nodeNeighborBaseProb=1, nodeNeighborStdDev=0.2, 
                connectedThreshold=0.05, intra_group_prob=0.09, inter_group_prob=0.005, repulsion_factor=0.34):
    
    if nodeAmount % groupsAmount != 0:
        print("Node amount must be divisible by groups amount")
        return
    
    # Determine the GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random generator and move data to GPU if available
    rng = torch.Generator(device).manual_seed(torch.seed())
    node_per_group = nodeAmount // groupsAmount

    # Create data on the GPU
    data = torch.zeros((groupsAmount, node_per_group, nodeDim), dtype=torch.float32, device=device)
    outliers = []
    all_group_averages = []

    # Set the first group's seed point randomly
    data[0, 0] = torch.rand(nodeDim, generator=rng, device=device)

    # Generate subsequent group seed points with randomness while ensuring minimum distance
    for group in range(1, groupsAmount):
        successful = False

        # Attempt to generate a new seed point with sufficient distance from previous seeds
        for i in range(10):  # Limit attempts to avoid infinite loops
            candidate_position = torch.rand(nodeDim, generator=rng, device=device)
            distances = torch.linalg.norm(candidate_position - data[:group, 0], dim=1)
            if torch.min(distances) > repulsion_factor:
                data[group, 0] = candidate_position
                successful = True
                break

        # If we couldn't find a suitable candidate after 100 attempts, adjust to the closest valid position
        if not successful:
            farthest_seed = data[:group, 0] + torch.randn_like(data[:group, 0]) * repulsion_factor
            data[group, 0] = farthest_seed.mean(dim=0)

    for group in range(groupsAmount):
        # if group == 0:
        #     data[group, 0] = torch.rand(nodeDim, generator=rng, device=device)
        # else:
        #     # Find position far enough from other group seeds
        #     while True:
        #         candidate_position = torch.rand(nodeDim, generator=rng, device=device)
        #         min_distance = torch.min(torch.linalg.norm(candidate_position - data[:group, 0], dim=1))
        #         if min_distance > repulsion_factor:
        #             data[group, 0] = candidate_position
        #             break

        group_average = data[group, 0].clone()

        for node in range(1, node_per_group):
            # Running average update every 10% of nodes
            # print("YAHURR")
            # print(node)
            # print(node_per_group)
            if node_per_group >= 10:
                if node % (node_per_group // 10) == 0:
                    group_average = torch.mean(data[group, :node], dim=0)

            p = torch.rand(1, generator=rng, device=device).item()

            if p < nodeNeighborBaseProb:
                data[group, node] = torch.normal(mean=group_average, std=nodeNeighborStdDev, generator=rng)
            else:
                outliers.append((group, node))
                data[group, node] = torch.normal(mean=group_average, std=nodeNeighborStdDev, generator=rng)

        all_group_averages.append(group_average)

    # Create outliers using global average
    average_array = torch.mean(torch.stack(all_group_averages), dim=0)
    for group, node in outliers:
        data[group, node] = torch.normal(mean=group_average, std=nodeNeighborStdDev, generator=rng)

    # Combine all nodes into one array
    all_nodes = data.view(nodeAmount, nodeDim)

    # Create adjacency matrix based on distances and group probabilities
    adjacency_matrix = torch.zeros((nodeAmount, nodeAmount), dtype=torch.int32, device=device)
    
    # Compute pairwise distances using GPU (batch computation)
    distances = torch.cdist(all_nodes, all_nodes, p=2)  # This will generate a matrix of distances

    # Compute group indices
    group_indices = torch.arange(nodeAmount, device=device) // node_per_group
    intra_mask = group_indices.unsqueeze(0) == group_indices.unsqueeze(1)

    # Calculate probabilities for intra and inter group links
    probabilities = torch.where(intra_mask, intra_group_prob / (1 + distances), inter_group_prob / (1 + distances))
    probabilities = torch.where(distances <= connectedThreshold, probabilities * 2, probabilities)

    # Randomize connections based on probabilities (efficiently, on GPU)
    random_probs = torch.rand((nodeAmount, nodeAmount), generator=rng, device=device)
    adjacency_matrix = (random_probs < probabilities).int()

    # Make the adjacency matrix symmetric
    adjacency_matrix = torch.triu(adjacency_matrix, diagonal=1)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T

    # Create labels
    labels = torch.arange(groupsAmount, device=device).repeat_interleave(node_per_group)

    # Shuffle nodes and their labels
    shuffle_indices = torch.randperm(nodeAmount, generator=rng, device=device)
    shuffled_all_nodes = all_nodes[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]
    shuffled_adj = adjacency_matrix[shuffle_indices][:, shuffle_indices]

    return data, shuffled_adj, shuffled_all_nodes, shuffled_labels

if __name__ == "__main__":
    groupsAmount = 2
    nodeAmount = 200
    iterations = 1
    # data, adj, all_nodes, labels = makeDataSet(groupsAmount=2, intra_group_prob=0.1, inter_group_prob=0.01)
    # data, adj, all_nodes, labels = makeDataSetCUDA(nodeNeighborStdDev=0.2, nodeNeighborBaseProb = 1, nodeAmount=nodeAmount, groupsAmount=groupsAmount, repulsion_factor=0.5)
    data, adj, all_nodes, labels = makeDataSetCUDA(groupsAmount=groupsAmount, nodeAmount=nodeAmount, nodeNeighborStdDev=2)

    # Node each uniform random
    # Different types of distribution
    # Constraint nodes
    

    # # Time the execution of makeDataSet
    # start_time = time.time()
    # for i in tqdm(range(iterations)):
    #     data, adj, all_nodes, labels = makeDataSet(groupsAmount=groupsAmount, nodeAmount=nodeAmount)
    # makeDataSet_time = time.time() - start_time

    # # Time the execution of makeDataSetOLD
    # start_time = time.time()
    # for i in tqdm(range(iterations)):
    #     data_old, adj_old, all_nodes_old, labels_old = makeDataSetOLD(groupsAmount=groupsAmount, nodeAmount=nodeAmount)
    # makeDataSetOLD_time = time.time() - start_time

    # start_time = time.time()
    # for i in tqdm(range(iterations)):
    #     data_old, adj_old, all_nodes_old, labels_old = makeDataSetCUDA(groupsAmount=groupsAmount, nodeAmount=nodeAmount)
    # makeDataSetCUDA_time = time.time() - start_time
    

    # # Print the execution times
    # print(f"makeDataSet execution time: {makeDataSet_time:.4f} seconds")
    # print(f"makeDataSetOLD execution time: {makeDataSetOLD_time:.4f} seconds")
    # print(f"makeDataSetCUDA execution time: {makeDataSetCUDA_time:.4f} seconds")
    plot_dataset(data, adj, all_nodes, labels)
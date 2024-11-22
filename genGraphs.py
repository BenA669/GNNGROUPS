import torch
from makeDataset import makeDataSetCUDA, plot_dataset
from tqdm import tqdm

def generate_and_save_graphs(num_graphs=10000, groupsAmount=3, nodeAmount=300):
    graphs = []
    for _ in tqdm(range(num_graphs)):
        data, adj, all_nodes, labels = makeDataSetCUDA(groupsAmount=groupsAmount, nodeAmount=nodeAmount)
        graphs.append((data, adj, all_nodes, labels))
    
    torch.save(graphs, '{}_groups_{}_nodes_pregenerated_graphs.pt'.format(groupsAmount, nodeAmount))

    graphs_validation = []
    for _ in tqdm(range(num_graphs)):
        data, adj, all_nodes, labels = makeDataSetCUDA(groupsAmount=groupsAmount, nodeAmount=nodeAmount)
        graphs_validation.append((data, adj, all_nodes, labels))
    
    torch.save(graphs_validation, '{}_groups_{}_nodes_pregenerated_graphs_validation.pt'.format(groupsAmount, nodeAmount))

def sample_and_display_graphs(num_samples=5, file_path='3_groups_300_nodes_pregenerated_graphs_validation.pt'):
    # Load the pregenerated graphs
    graphs = torch.load(file_path)
    
    # Ensure num_samples does not exceed the number of available graphs
    num_samples = min(num_samples, len(graphs))
    
    # Randomly sample graphs
    sampled_graphs = torch.utils.data.random_split(graphs, [num_samples, len(graphs) - num_samples])[0]
    
    # Display each sampled graph
    for data, adj, all_nodes, labels in sampled_graphs:
        plot_dataset(data, 2, adj, all_nodes, labels)


if __name__ == "__main__":
    # sample_and_display_graphs(3)
    generate_and_save_graphs()
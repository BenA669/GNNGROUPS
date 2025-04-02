import torch
import argparse
from makeDataset import plot_dataset


def sample_and_display_graphs(num_samples=5, file_path='300_nodes_pregenerated_graphs.pt'):
    # Load the pregenerated graphs
    graphs = torch.load(file_path)
    
    # Ensure num_samples does not exceed the number of available graphs
    num_samples = min(num_samples, len(graphs))
    
    # Randomly sample graphs
    sampled_graphs = torch.utils.data.random_split(graphs, [num_samples, len(graphs) - num_samples])[0]
    
    # Display each sampled graph
    for data, adj, all_nodes, labels in sampled_graphs:
        plot_dataset(data, adj, all_nodes, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GCN model.')
    parser.add_argument('--i', type=int, default=1000, help='Number of samples')
    parser.add_argument('--m', type=str, default=1000, help='Dataset')
    args = parser.parse_args()

    sample_and_display_graphs(num_samples=args.i, file_path=args.m)
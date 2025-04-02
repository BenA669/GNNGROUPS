from model import GCN, ClusterPredictor
from evaluate import eval, InfoNCELoss
from Dynamic.makeEpisode import makeDatasetDynamic 
import torch
import torch.optim as optim
import torch.nn.functional as F
import statistics
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# Define constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
EPOCHS = 10000
EPOCH_UPDATE = 1000
LR = 0.001
WEIGHT_DECAY = 5e-4
TIME_STEPS = 20
THRESHOLD = 0.5  # Distance threshold for connecting nodes
HIDDEN_DIM1 = 64
OUTPUT_DIM = 16
LOSS_MEMORY_SIZE = 50
MAX_NODES = 64  # Define a maximum number of nodes to pad to

class DynamicGraphDataset(Dataset):
    def __init__(self, time_steps, node_amt, group_amt, std_dev, speed_min, speed_max, intra_prob, inter_prob, threshold):
        super(DynamicGraphDataset, self).__init__()
        self.time_steps = time_steps
        self.node_amt = node_amt
        self.group_amt = group_amt
        self.std_dev = std_dev
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.intra_prob = intra_prob
        self.inter_prob = inter_prob
        self.threshold = threshold
        # Generate the dynamic dataset
        self.all_positions, self.adj_matrices, self.all_positions_cpu, self.adj_matrices_cpu = makeDatasetDynamic(
            node_amt=self.node_amt,
            group_amt=self.group_amt,
            std_dev=self.std_dev,
            speed_min=self.speed_min,
            speed_max=self.speed_max,
            time_steps=self.time_steps,
            intra_prob=self.intra_prob,
            inter_prob=self.inter_prob
        )
        self.labels = self.get_labels()

    def get_labels(self):
        # Assuming the third column in positions represents group labels
        initial_positions = self.all_positions_cpu[0]
        node_groups = initial_positions[:, 2].long()
        return node_groups

    def __len__(self):
        return self.node_amt  # Each node can be a source node for an episode

    def __getitem__(self, idx):
        # For a given source node, extract its 2-hop neighborhood across all timesteps
        selected_nodes_all_timesteps = set()
        labels = self.labels[idx].item()
        
        # First pass: collect all selected_nodes across all timesteps
        for t in range(self.time_steps):
            positions = self.all_positions_cpu[t]  # (node_amt, 3)
            coords = positions[:, :2]  # x and y coordinates (on CPU)

            # Compute distance matrix using CPU
            dist_matrix = squareform(pdist(coords.cpu().numpy()))
            dist_matrix = torch.tensor(dist_matrix, device='cpu')  # Ensure on CPU

            # Compute adjacency based on threshold
            adj = (dist_matrix < self.threshold).float()
            adj.fill_diagonal_(0)  # Remove self-loops

            # Extract 2-hop neighborhood
            # First, find direct neighbors
            direct_neighbors = adj[idx] > 0
            direct_indices = torch.nonzero(direct_neighbors, as_tuple=False).view(-1)  # On CPU

            # Then, find neighbors of neighbors
            if direct_indices.numel() > 0:
                two_hop_neighbors = adj[direct_indices].sum(dim=0) > 0
                two_hop_indices = torch.nonzero(two_hop_neighbors, as_tuple=False).view(-1)  # On CPU
            else:
                two_hop_indices = torch.tensor([], dtype=torch.long, device='cpu')  # Empty tensor on CPU

            # Combine source, direct, and two-hop neighbors
            source_node = torch.tensor([idx], device='cpu')  # Ensure on CPU
            selected_nodes = torch.unique(torch.cat([
                source_node,
                direct_indices,
                two_hop_indices
            ])).long()  # On CPU

            # Update the union set with selected nodes from this timestep
            selected_nodes_all_timesteps.update(selected_nodes.tolist())

        # Now, all selected_nodes across all timesteps
        selected_nodes_union = sorted(selected_nodes_all_timesteps)  # sorted list for consistency
        selected_nodes_union_tensor = torch.tensor(selected_nodes_union, device='cpu').long()

        # Limit to MAX_NODES to prevent excessively large subgraphs
        if len(selected_nodes_union) > MAX_NODES:
            selected_nodes_union = selected_nodes_union[:MAX_NODES]
            selected_nodes_union_tensor = torch.tensor(selected_nodes_union, device='cpu').long()

        # Now, for each timestep, extract subgraphs based on the union set
        episodes = []
        for t in range(self.time_steps):
            positions = self.all_positions_cpu[t]  # (node_amt, 3)
            coords = positions[:, :2]  # x and y coordinates (on CPU)

            # Compute distance matrix using CPU
            dist_matrix = squareform(pdist(coords.cpu().numpy()))
            dist_matrix = torch.tensor(dist_matrix, device='cpu')  # Ensure on CPU

            # Compute adjacency based on threshold
            adj = (dist_matrix < self.threshold).float()
            adj.fill_diagonal_(0)  # Remove self-loops

            # Create sub-adjacency matrix based on selected_nodes_union
            sub_adj = adj[selected_nodes_union_tensor][:, selected_nodes_union_tensor]

            # Get sub-coordinates
            sub_coords = coords[selected_nodes_union_tensor]

            # Get sub-labels
            sub_labels = self.labels[selected_nodes_union_tensor]

            episode = {
                'coords': sub_coords.float(),  # (max_N, 2)
                'adj': sub_adj.float(),        # (max_N, max_N)
                'labels': sub_labels            # (max_N,)
            }
            episodes.append(episode)

        # Stack episodes across timesteps
        # Shape: (time_steps, max_N, 2) for coords
        #         (time_steps, max_N, max_N) for adj
        #         (time_steps, max_N) for labels
        stacked_coords = torch.stack([ep['coords'] for ep in episodes], dim=0)  # (T, N, 2)
        stacked_adj = torch.stack([ep['adj'] for ep in episodes], dim=0)        # (T, N, N)
        stacked_labels = torch.stack([ep['labels'] for ep in episodes], dim=0)  # (T, N)

        return stacked_coords, stacked_adj, stacked_labels, labels

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized subgraphs by padding.
    
    Args:
        batch: List of tuples, each containing:
            - coords: Tensor of shape (TIME_STEPS, N_i, 2)
            - adjs: Tensor of shape (TIME_STEPS, N_i, N_i)
            - labels: Tensor of shape (TIME_STEPS, N_i)
            - group_label: Scalar label for the sample

    Returns:
        padded_coords: Tensor of shape (B, TIME_STEPS, max_N, 2)
        padded_adjs: Tensor of shape (B, TIME_STEPS, max_N, max_N)
        padded_labels: Tensor of shape (B, TIME_STEPS, max_N)
        group_labels: Tensor of shape (B,)
    """
    coords, adjs, labels, group_labels = zip(*batch)
    
    # Determine the maximum number of nodes in the batch
    max_nodes = max(sample_coords.size(1) for sample_coords in coords)
    
    # Initialize padded tensors
    batch_size = len(batch)
    T = TIME_STEPS
    padded_coords = torch.zeros(batch_size, T, max_nodes, 2, device=DEVICE)
    padded_adjs = torch.zeros(batch_size, T, max_nodes, max_nodes, device=DEVICE)
    padded_labels = torch.full((batch_size, T, max_nodes), -1, dtype=torch.long, device=DEVICE)  # -1 as padding label
    
    # Populate the padded tensors
    for i in range(batch_size):
        sample_size = coords[i].size(1)
        padded_coords[i, :, :sample_size, :] = coords[i].to(DEVICE)
        padded_adjs[i, :, :sample_size, :sample_size] = adjs[i].to(DEVICE)
        padded_labels[i, :, :sample_size] = labels[i].to(DEVICE)
    
    # Convert group_labels to a tensor
    group_labels = torch.tensor(group_labels, device=DEVICE)
    
    return padded_coords, padded_adjs, padded_labels, group_labels

def main():
    # Initialize dataset and dataloader
    dataset = DynamicGraphDataset(
        time_steps=TIME_STEPS,
        node_amt=400,
        group_amt=4,
        std_dev=1,
        speed_min=0.01,
        speed_max=0.5,
        intra_prob=0.05,
        inter_prob=0.001,
        threshold=THRESHOLD
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging; increase later as needed
    )

    # Initialize models
    model = GCN(input_dim=2, hidden_dim1=HIDDEN_DIM1, output_dim=OUTPUT_DIM).to(DEVICE)
    cluster_model = ClusterPredictor(OUTPUT_DIM).to(DEVICE)

    # Define optimizers
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cluster_optimizer = optim.Adam(cluster_model.parameters(), lr=LR)

    # Initialize loss memory
    previous_losses = []
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        cluster_model.train()
        epoch_loss = 0

        for batch in dataloader:
            coords, adjs, labels, group_labels = batch  # All tensors on DEVICE

            # Forward pass through each timestep
            batch_loss = 0

            for t in range(TIME_STEPS):
                batch_coords_t = coords[:, t, :, :]  # (B, max_N, 2)
                batch_adjs_t = adjs[:, t, :, :]      # (B, max_N, max_N)
                batch_labels_t = labels[:, t, :]     # (B, max_N)

                # Flatten the batch for GCN input
                # Reshape to (B*max_N, 2)
                B, N, F = batch_coords_t.shape
                flat_coords = batch_coords_t.reshape(B * N, F)  # (B*max_N, 2)
                flat_adjs = batch_adjs_t.reshape(B * N, N)      # (B*max_N, N)
                flat_labels = batch_labels_t.reshape(B * N)     # (B*max_N,)

                # Forward pass
                output = model(flat_coords, flat_adjs)  # (B*max_N, OUTPUT_DIM)

                # Create mask to ignore padding labels (-1)
                mask = flat_labels != -1
                valid_labels = flat_labels[mask]
                valid_output = output[mask]

                # Calculate InfoNCE Loss only on valid entries
                if valid_output.numel() > 0:
                    loss = InfoNCELoss(valid_output, valid_labels)
                    batch_loss += loss
                else:
                    # If no valid labels, skip this timestep
                    continue

            # Backpropagate
            if TIME_STEPS > 0:
                total_loss = batch_loss / TIME_STEPS  # Average over timesteps
                total_loss.backward()

                # Update weights
                optimizer.step()
                optimizer.zero_grad()

                # Record loss
                epoch_loss += total_loss.item()

        # Record average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        previous_losses.append(avg_epoch_loss)
        if len(previous_losses) > LOSS_MEMORY_SIZE:
            previous_losses.pop(0)

        if epoch % EPOCH_UPDATE == 0:
            print(f'Epoch {epoch}, Loss Average: {statistics.mean(previous_losses)}')
            # Optionally, perform evaluation
            # print("Evaluation: {}".format(eval(model, cluster_model, 1000, graphs_validation)))

    # Save models
    torch.save(model.state_dict(), 'gcn_model_dynamic.pth')
    print("Model saved as 'gcn_model_dynamic.pth'")
    torch.save(cluster_model.state_dict(), 'cluster_model_dynamic.pth')
    print("Cluster model saved as 'cluster_model_dynamic.pth'")

if __name__ == '__main__':
    main()

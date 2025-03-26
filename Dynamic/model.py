import torch
import torch.nn as nn
from GraphDataset import GCNDataset, collate_fn
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def tensor_to_edge_index(adj):
    """
    Convert an NxN adjacency matrix (PyTorch tensor) to an edge index tensor of shape [2, num_edges].

    Parameters:
        adj (torch.Tensor): A square (NxN) tensor representing the adjacency matrix.
        
    Returns:
        torch.Tensor: A tensor of shape [2, num_edges] where the first row contains the source nodes
                      and the second row contains the target nodes of the edges.
    """
    # Ensure the tensor is square
    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("The input tensor must be a square (NxN) matrix.")

    # Find the indices of nonzero entries (which represent edges)
    # as_tuple=True returns separate coordinate tensors for rows and columns.
    src, dst = torch.nonzero(adj, as_tuple=True)
    
    # Stack them to form an edge_index tensor of shape [2, num_edges]
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


class TemporalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, num_timesteps, hidden_dim=64):
        super(TemporalGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"IS CUDA AVALAIBLE: {torch.cuda.is_available()}")
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, big_batch_adjacency, ego_mask, eval=False):
        x_out = []

        # In eval mode?
        if eval:
            B = 1
            x_ego_mask = ego_mask
            max_nodes = ego_mask.size(dim=1)
        else:
            B = len(ego_mask)
            # [Batch, Time, Nodes] -> [Time, Batch*Nodes]
            x_ego_mask = ego_mask.permute(1, 0, 2).reshape(20, -1) # Prepare ego_mask to mask x
            max_nodes = ego_mask.size(dim=2)

        x_placeholder = torch.zeros(self.num_timesteps, max_nodes*B, self.hidden_dim).to(self.device)
        for t in range(self.num_timesteps):
            x_t = x[t]                      # Get features at timestamp t
            a_t = big_batch_adjacency[t]    # Get adjacency at timestamp t
            ego_mask_t = x_ego_mask[t]      # Get ego mask at timestamp t
            
            # Post Pad:
            x_t = x_t[ego_mask_t]                   # Mask features
            a_t = a_t[ego_mask_t][:, ego_mask_t]    # Mask adjacency

            # Pre Pad:
            # x_t = x_t.masked_fill(~x_ego_mask.any(dim=0).unsqueeze(1).to(self.device), -5000)     # Mask features
            
            # mask = ~ego_mask_t.unsqueeze(1) & ~ego_mask_t.unsqueeze(0)  # Create a mask for adjacency
            # a_t = a_t.masked_fill(mask.to(self.device), False)

            e_t = tensor_to_edge_index(a_t)         # Convert adjacency matrix to edge index (2, Y)
            
            x_t = self.gcn1(x_t, e_t)               # Pass masked features and adjacency
            x_t = torch.relu(x_t)
            x_t = self.gcn2(x_t, e_t)

            # Post Pad
            ego_idx = torch.nonzero(ego_mask_t).flatten().to(self.device)
            x_placeholder[t, ego_idx] = x_t         # Insert embeddings into their corresponding place in the global matrix (Padding)

            # Pre Pad
            # x_placeholder[t] = x_t

        # _, (h_n, _) = self.lstm(x_placeholder)
        # x_out = h_n[-1]  # Get last layer's hidden state -> [batch, node_amt * hidden_dim]

        
        # x_out = x_out.view(B, 400, -1) # [batch, node_amt * hidden_dim] -> [batch_size, num_nodes, output_dim]
        
        # x_out = self.fc(x_out)  

        # Compute sequence lengths for each node (each column in x_placeholder is a node's time series)
        if eval:
            # In eval mode, assume all nodes are present at every timestep.
            lengths = torch.full((B * max_nodes,), self.num_timesteps, dtype=torch.long, device=self.device)
        else:
            # For each node in the flattened batch, count valid timesteps.
            # x_ego_mask has shape [num_timesteps, B * max_nodes]; summing over time gives the lengths.
            lengths = x_ego_mask.sum(dim=0)

        lengths = torch.clamp(lengths, min=1)

        # Pack the padded sequence (note: lengths must be on CPU)
        packed_input = pack_padded_sequence(x_placeholder, lengths.cpu(), enforce_sorted=False)

        # Pass the packed sequence to the LSTM.
        # The LSTM will only process the valid (non-padded) timesteps for each node.
        packed_output, (h_n, _) = self.lstm(packed_input)
        
        # h_n[-1] gives the hidden state of the last LSTM layer for each node.
        x_out = h_n[-1]  # Shape: [B * max_nodes, hidden_dim]

        # Reshape the output to [B, max_nodes, hidden_dim]
        x_out = x_out.view(B, max_nodes, -1)
        
        # Optionally, apply a fully connected layer.
        x_out = self.fc(x_out)
        return x_out

if __name__ == '__main__':
    # Number of nodes, timesteps, and features
    num_nodes = 100
    num_timesteps = 20
    input_dim = 2
    output_dim = 16

    # Example graph data (node features)
    # x = torch.rand(4, num_timesteps, num_nodes, input_dim)  # [batch_size, num_nodes, num_timesteps, input_dim]

    # # Generate a list of adjacency lists for each timestep
    # edge_indices = [torch.randint(0, num_nodes, (2, 500)) for _ in range(num_timesteps)]


    # Define the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TemporalGCN(input_dim, output_dim, num_nodes, num_timesteps).to(device)

    # # Forward pass
    # print("INPUT SHAPE 1: {}".format(x.shape))
    # output = model(x, edge_indices)
    # print(output.shape)  # Expected shape: [batch_size, num_nodes, output_dim]


    dataset = GCNDataset('test_data_Ego.pt')

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions'] # batch, timestamp, node_amt, 3
        edge_indices = batch['edge_indices']
        ego_mask_batch = batch['ego_mask_batch']
        big_batch_edges = batch['big_batch_edges']
        big_batch_positions = batch['big_batch_positions']
        big_batch_ego_edges = batch['big_batch_ego_edges']
        big_batch_adjacency = batch['big_batch_adjacency']

        groups = positions[:, 0, :, 2]

        emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)

        break

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, num_timesteps, hidden_dim=64):
        super(TemporalGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_indices):
        """
        Args:
            x: Node features, shape [batch_size, num_nodes, num_timesteps, input_dim]
            edge_indices: List of edge_index tensors for each timestep, each shape [2, num_edges_t]
        Returns:
            Output predictions for each node
        """
        batch_size = x.size(0)
        
        # Reshape for processing: [batch_size * num_nodes, num_timesteps, input_dim]
        x = x.view(-1, self.num_timesteps, x.size(-1))
        
        # Apply GCN layers with time-dependent adjacency matrices
        x_out = []
        for t in range(self.num_timesteps):
            x_t = x[:, t, :]  # Features at timestep t
            
            # Use time-specific adjacency matrix
            edge_index_t = edge_indices[t]  # [2, num_edges_t]
            
            x_t = self.gcn1(x_t, edge_index_t)
            x_t = torch.relu(x_t)
            x_t = self.gcn2(x_t, edge_index_t)
            x_out.append(x_t)
        
        # Combine temporal features: [batch_size * num_nodes, num_timesteps, hidden_dim]
        x_out = torch.stack(x_out, dim=1)
        
        # Apply LSTM: [batch_size * num_nodes, num_timesteps, hidden_dim] -> [batch_size * num_nodes, hidden_dim]
        _, (h_n, _) = self.lstm(x_out)
        x_out = h_n[-1]
        
        # Reshape back to [batch_size, num_nodes, hidden_dim]
        x_out = x_out.view(batch_size, self.num_nodes, -1)
        
        # Final output layer
        x_out = self.fc(x_out)  # [batch_size, num_nodes, output_dim]
        
        return x_out


# Number of nodes, timesteps, and features
num_nodes = 100
num_timesteps = 10
input_dim = 16
output_dim = 4

# Example graph data (node features)
x = torch.rand(1, num_nodes, num_timesteps, input_dim)  # [batch_size, num_nodes, num_timesteps, input_dim]

# Generate a list of adjacency lists for each timestep
edge_indices = [torch.randint(0, num_nodes, (2, 500)) for _ in range(num_timesteps)]

# Define the model
model = TemporalGCN(input_dim, output_dim, num_nodes, num_timesteps)

# Forward pass
output = model(x, edge_indices)
print(output.shape)  # Expected shape: [batch_size, num_nodes, output_dim]

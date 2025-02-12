import torch
import torch.nn as nn
from datasettest import GCNDataset, collate_fn
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, num_timesteps, hidden_dim=64):
        super(TemporalGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        

    def forward(self, x, edge_indices, ego_mask):
        
        x_out = []
        # print("B? : {}".format(len(ego_mask)))
        B = len(ego_mask)
        # print("X_OUT shape0 : {}".format(x[0].shape))
        for t in range(self.num_timesteps):
            x_t = x[t]
            e_t = edge_indices[t]
            x_t = self.gcn1(x_t, e_t)
            x_t = torch.relu(x_t)
            x_t = self.gcn2(x_t, e_t)
            x_out.append(x_t)

        x_out = torch.stack(x_out, dim=0)
        # print("X_OUT shape1 : {}".format(x_out.shape))

        
        mask = (~ego_mask.transpose(0, 1).reshape(20, -1, 1)).float().to(self.device)
        # Zero out the masked positions:
        x_out_masked = x_out * mask
        _, (h_n, _) = self.lstm(x_out_masked)
        x_out = h_n[-1]  # Get last layer's hidden state -> [batch, node_amt * hidden_dim]

        x_out = x_out.view(B, 400, -1) # [batch, node_amt * hidden_dim] -> [batch_size, num_nodes, output_dim]
        # print("X_OUT shape 2: {}".format(x_out.shape))
        x_out = self.fc(x_out)  
        
        print("one pass done")
        return x_out
        
        #####
        '''
        # x = [batch, timesteps, node_amt, 2]
        # ego_adjacency = [batch, timesteps, node_amt, node_amt]
        # edge_indices = [batch, timestep, 2, node_amt]
        batch_size, timesteps_amt, node_amt_max, _ = x.shape

        x_out = []
        for t in range(timesteps_amt):
            x_t = x[:, t, :, :]
            x_t = self.gcn1(x_t, edge_indices[t])
            x_t = torch.relu(x_t)
            x_t = self.gcn2(x_t, edge_indices[t])
            x_out.append(x_t)

        x_out = torch.stack(x_out, dim=1)  # [timesteps, batch, node_amt, hidden_dim]
        print("XOUT:{}".format(x_out.shape))
        
        # Merge batch and node dimensions: new shape becomes [timesteps, batch * node_amt, hidden_dim]
        x_out = x_out.view(timesteps_amt, batch_size * node_amt_max, -1)

        _, (h_n, _) = self.lstm(x_out)  
        x_out = h_n[-1]  # Get last layer's hidden state -> [batch, node_amt * hidden_dim]

        x_out = x_out.view(batch_size, node_amt_max, -1) # [batch, node_amt * hidden_dim] -> [batch_size, num_nodes, output_dim]
        x_out = self.fc(x_out)  
        return x_out
        '''




        ##########################
        """
        batch_size = x.size(0)
        num_nodes_Dyn = x.size(1)

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
        x_out = x_out.view(batch_size, num_nodes_Dyn, -1)
        
        # Final output layer
        x_out = self.fc(x_out)  # [batch_size, num_nodes, output_dim]
        
        return x_out
        """

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

        groups = positions[:, 0, :, 2]
        print("Groups shape: {}".format(groups.shape))

        emb = model(big_batch_positions, big_batch_edges, ego_mask_batch)

        break

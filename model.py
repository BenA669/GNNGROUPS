import torch
import torch.nn as nn
from datasetEpisode import GCNDataset, collate_fn
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import configparser

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

# class TimeSeriesNN(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """
#     def __init__(self, size_in, size_out):
#         super().__init__()
#         self.size_in, self.size_out = size_in, size_out
#         weights = torch.Tensor(size_out, size_in)
#         self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
#         bias = torch.Tensor(size_out)
#         self.bias = nn.Parameter(bias)

#         # initialize weights and biases
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in)
#         nn.init.uniform_(self.bias, -bound, bound)  # bias init

#     def forward(self, x):
#         w_times_x= torch.mm(x, self.weights.t())
#         return torch.add(w_times_x, self.bias)  # w times x + b

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(TemporalGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"IS CUDA AVALAIBLE: {torch.cuda.is_available()}")
        # self.num_nodes = num_nodes
        # self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolution layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(hidden_dim, output_dim)

        self.attention = nn.Linear(output_dim, 1)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, batch, eval=False):
        ego_mask = batch['ego_mask_batch'] # Shape: (Batch, Timestep, Node Amt)
        x = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batched_adjacency_pruned']

        num_timesteps = x.shape[0]

        x_out = []

        # In eval mode?
        if eval:
            B = 1
            x_ego_mask = ego_mask
            max_nodes = ego_mask.size(dim=1)
        else:
            B = len(ego_mask)
            # [Batch, Time, Nodes] -> [Time, Batch*Nodes]
            x_ego_mask = ego_mask.permute(1, 0, 2).reshape(num_timesteps, -1) # Prepare ego_mask to mask x
            max_nodes = ego_mask.size(dim=2)

        x_placeholder = torch.zeros(num_timesteps, max_nodes*B, self.hidden_dim).to(self.device)

        for t in range(num_timesteps):
            x_t = x[t]                      # Get features at timestamp t
            a_t = big_batch_adjacency[t]    # Get adjacency at timestamp t
            ego_mask_t = x_ego_mask[t]      # Get ego mask at timestamp t
            
            # Post Pad:
            x_t_m = x_t[ego_mask_t]                   # Mask features
            a_t_m = a_t[ego_mask_t][:, ego_mask_t]    # Mask adjacency

            e_t = tensor_to_edge_index(a_t_m)         # Convert adjacency matrix to edge index (2, Y)
            
            x_t_g1 = self.gcn1(x_t_m, e_t)               # Pass masked features and adjacency
            x_t_r = torch.relu(x_t_g1)
            x_t_g2 = self.gcn2(x_t_r, e_t)

            # Post Pad
            ego_idx = torch.nonzero(ego_mask_t).flatten().to(self.device)
            x_placeholder[t, ego_idx] = x_t_g2         # Insert embeddings into their corresponding place in the global matrix (Padding)

            # Pre Pad
            # x_placeholder[t] = x_t

        # Rearrange the placeholder for LSTM processing.
        # Currently: [T, B * max_nodes, hidden_dim] -> reshape to (B*num_nodes, T, hidden_dim)
        x_placeholder = x_placeholder.transpose(0, 1)  # Now shape: (B * max_nodes, T, hidden_dim)
        

        lstm_out, (h_n, _) = self.lstm(x_placeholder)
        

        attn_scores = self.attention(lstm_out)  # (B*num_nodes, T, 1)
        attn_scores = attn_scores.squeeze(-1)   # (B*num_nodes, T)


        # Compute attention weights.
        attn_weights = F.softmax(attn_scores, dim=1)  # (B*num_nodes, T)

        # Compute the context vector as a weighted sum of LSTM outputs (over the time dimension).
        # Expand attn_weights and multiply elementwise with lstm_out.
        attn_weights = attn_weights.unsqueeze(-1)    # (B*num_nodes, T, 1)
        x_out = torch.sum(attn_weights * lstm_out, dim=1)  # (B*num_nodes, hidden_dim)
        
        # Reshape the context vector to (B, max_nodes, hidden_dim)
        x_out = x_out.view(B, max_nodes, self.output_dim)
        
        # print(context.shape)
        # # Final fully connected layer to get output embeddings.
        # x_out = self.fc(context)   # (B, max_nodes, output_dim)

        # print(x_out.shape)
        return x_out

        # x_out = h_n[-1]  # Get last layer's hidden state -> [batch, node_amt * hidden_dim]
        # x_out = x_out.view(B, 100, -1) # [batch, node_amt * hidden_dim] -> [batch_size, num_nodes, output_dim]
        
        # x_out = self.fc(x_out)  

        # return x_out

        # Compute sequence lengths for each node (each column in x_placeholder is a node's time series)
        # if eval:
        #     # In eval mode, assume all nodes are present at every timestep.
        #     lengths = torch.full((B * max_nodes,), self.num_timesteps, dtype=torch.long, device=self.device)
        # else:
        #     # For each node in the flattened batch, count valid timesteps.
        #     # x_ego_mask has shape [num_timesteps, B * max_nodes]; summing over time gives the lengths.
        #     lengths = x_ego_mask.sum(dim=0)

        # lengths = torch.clamp(lengths, min=1)

        # # Pack the padded sequence (note: lengths must be on CPU)
        # packed_input = pack_padded_sequence(x_placeholder, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass the packed sequence to the LSTM.
        # The LSTM will only process the valid (non-padded) timesteps for each node.
        # packed_output, (h_n, _) = self.lstm(packed_input)

        # print("Packed Output Shape: ")
        # print(packed_output)

        # Get padded LSTM outputs.
        # lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)  # (B*num_nodes, T, hidden_dim)

        # # h_n[-1] gives the hidden state of the last LSTM layer for each node.
        # x_out = h_n[-1]  # Shape: [B * max_nodes, hidden_dim]

        # # Reshape the output to [B, max_nodes, hidden_dim]
        # x_out = x_out.view(B, max_nodes, -1)
        
        # # Optionally, apply a fully connected layer.
        # x_out = self.fc(x_out)
        # return x_out

         # Apply attention: compute a score for each timestep of each node.
        # Here we project each hidden state to a scalar score.
        # Note: lstm_out may have padding positions; you can mask them out if needed.
        # attn_scores = self.attention(lstm_out)  # (B*num_nodes, T, 1)
        # attn_scores = attn_scores.squeeze(-1)   # (B*num_nodes, T)
        
        # # For nodes with variable sequence lengths, it is beneficial to mask the padded positions.
        # # Create a mask from the sequence lengths.
        # max_T = lstm_out.size(1)
        # mask = torch.arange(max_T, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B*num_nodes, T)
        # # Set scores of padded positions to a large negative value before softmax.
        # attn_scores[~mask] = float("-inf")
        
        # # Compute attention weights.
        # attn_weights = F.softmax(attn_scores, dim=1)  # (B*num_nodes, T)
        
        # # Compute the context vector as a weighted sum of LSTM outputs (over the time dimension).
        # # Expand attn_weights and multiply elementwise with lstm_out.
        # attn_weights = attn_weights.unsqueeze(-1)    # (B*num_nodes, T, 1)
        # context = torch.sum(attn_weights * lstm_out, dim=1)  # (B*num_nodes, hidden_dim)
        
        # # Reshape the context vector to (B, max_nodes, hidden_dim)
        # context = context.view(B, max_nodes, self.hidden_dim)
        
        # # Final fully connected layer to get output embeddings.
        # x_out = self.fc(context)   # (B, max_nodes, output_dim)
        # return x_out
    
        

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Number of nodes, timesteps, and features
    num_nodes = int(config["dataset"]["nodes"])
    num_timesteps = int(config["dataset"]["timesteps"])
    input_dim = int(config["model"]["input_dim"])
    output_dim = int(config["model"]["output_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])

    dataset = GCNDataset(str(config["dataset"]["dataset_val"]))

    batch_size = int(config["training"]["batch_size"])


    # Example graph data (node features)
    # x = torch.rand(4, num_timesteps, num_nodes, input_dim)  # [batch_size, num_nodes, num_timesteps, input_dim]

    # # Generate a list of adjacency lists for each timestep
    # edge_indices = [torch.randint(0, num_nodes, (2, 500)) for _ in range(num_timesteps)]


    # Define the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TemporalGCN(input_dim, output_dim, hidden_dim).to(device)

    # # Forward pass
    # print("INPUT SHAPE 1: {}".format(x.shape))
    # output = model(x, edge_indices)
    # print(output.shape)  # Expected shape: [batch_size, num_nodes, output_dim]


    # dataset = GCNDataset('test_data_Ego_2hop.pt')
    

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions'] # batch, timestamp, node_amt, 3
        ego_mask_batch = batch['ego_mask_batch']
        big_batch_positions = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batch_adjacency']

        # emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)
        emb = model(batch)

        break

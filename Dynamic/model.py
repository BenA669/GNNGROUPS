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
        

    def forward(self, x, edge_indices, ego_mask, eval=False):
        # x = [Time, Batch*Nodes, (x,y)]
        # ego_mask = [Batch, Time, Nodes]
        
        # [Batch, Time, Nodes] -> [Time, Batch*Nodes]
        x_ego_mask = ego_mask.permute(1, 0, 2).reshape(20, -1) # Prepare ego_mask to mask x
        # x_ego_mask = [Time, Batch*Nodes]
        # print("Ego Mask Shape:{}".format(x_ego_mask.shape))
        # print("x Shape:{}".format(x.shape))

        x_out = []

        # In eval mode?
        if eval:
            B = 1
        else:
            B = len(ego_mask)


        # print("B? : {}".format(B))
        # print("X_OUT shape0 : {}".format(x[0].shape))
        for t in range(self.num_timesteps):
            x_t = x[t]
            e_t = edge_indices[t]
            ego_mask_t = x_ego_mask[t]
            
            x_t = x_t[ego_mask_t]
            
            print("x_t shape: {}".format(x_t.shape))
            print("e_t shape: {}".format(e_t.shape))

            exit()

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

        groups = positions[:, 0, :, 2]

        emb = model(big_batch_positions, big_batch_ego_edges, ego_mask_batch)

        break

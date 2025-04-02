from layers import GraphConvolution

import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_dim, hidden_dim1)
        self.gc2 = GraphConvolution(hidden_dim1, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, input, adj):
        x = F.relu(self.gc1(input, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)
    
class ClusterPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(ClusterPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)  # Output a single value for the number of clusters

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
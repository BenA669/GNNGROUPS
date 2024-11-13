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
        
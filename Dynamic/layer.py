# layer.py

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        input: [batch_size, num_nodes, in_features]
        adj: [batch_size, num_nodes, num_nodes]
        """
        device = input.device
        A_tilde = adj + torch.eye(adj.size(1), device=device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        
        degree = torch.sum(adj, dim=2)
        degree_matrix_inv_sqrt = torch.diag_embed(torch.pow(degree + 1e-6, -0.5))
        
        adj_normalized = torch.bmm(torch.bmm(degree_matrix_inv_sqrt, A_tilde), degree_matrix_inv_sqrt)
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj_normalized, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output

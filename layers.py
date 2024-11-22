import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = Parameter(torch.empty(in_features, out_features, device=self.device))

        if bias:
            # self.bias = Parameter(torch.FloatTensor(out_features, device=device))
            self.bias = Parameter(torch.empty(out_features, device=self.device))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        # input = H
        # adj = A
        eps=1e-6

        if str(self.device) == "cuda":
            if str(adj.device) == 'cpu':
                print("ADJ TO CUDA")
                adj = adj.to('cuda')
            if str(input.device) == 'cpu':
                print("INPUT TO CUDA")
                input = input.to('cuda')

        A_tilde = adj + torch.eye(adj.shape[0], device=self.device)


        degree = torch.sum(adj, dim=1)
        degree_matrix_inv_sqrt = torch.diag(torch.pow(degree + eps, -0.5))

        adj_normalized = torch.mm(degree_matrix_inv_sqrt, torch.mm(A_tilde, degree_matrix_inv_sqrt))
        # input = input.to(adj_normalized.dtype)
        output = torch.mm(adj_normalized, torch.mm(input, self.weight))

        return output
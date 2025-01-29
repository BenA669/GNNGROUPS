# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=2):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.gc_layers = nn.ModuleList()
        
        # Input layer
        self.gc_layers.append(GraphConvolution(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gc_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        # Output layer
        self.gc_layers.append(GraphConvolution(hidden_dim, embedding_dim))
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj):
        """
        x: [batch_size, num_nodes, input_dim]
        adj: [batch_size, num_nodes, num_nodes]
        """
        for i, gc in enumerate(self.gc_layers):
            x = gc(x, adj)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x  # [batch_size, num_nodes, embedding_dim]

class TemporalGCN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, embedding_dim=16, num_layers=2):
        super(TemporalGCN, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.temporal_fc = nn.Linear(embedding_dim * 20, embedding_dim)  # Assuming 20 timesteps

    def forward(self, x_seq, adj_seq):
        """
        x_seq: [batch_size, timesteps, num_nodes, input_dim]
        adj_seq: [batch_size, timesteps, num_nodes, num_nodes]
        """
        batch_size, timesteps, num_nodes, input_dim = x_seq.size()
        embeddings = []
        for t in range(timesteps):
            x = x_seq[:, t, :, :]  # [batch_size, num_nodes, input_dim]
            adj = adj_seq[:, t, :, :]  # [batch_size, num_nodes, num_nodes]
            embedding = self.encoder(x, adj)  # [batch_size, num_nodes, embedding_dim]
            embeddings.append(embedding)
        
        # Concatenate embeddings across timesteps
        embeddings = torch.cat(embeddings, dim=2)  # [batch_size, num_nodes, embedding_dim * timesteps]
        
        # Pass through a fully connected layer to reduce dimensionality
        embeddings = self.temporal_fc(embeddings)  # [batch_size, num_nodes, embedding_dim]
        
        return embeddings  # Final node embeddings

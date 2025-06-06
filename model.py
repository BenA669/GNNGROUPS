import torch
import torch.nn as nn
from datasetEpisode import GCNDataset, collate_fn
from torch_geometric.nn import GCNConv, GATConv
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import configparser
import math

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

class TrainOT(nn.Module):
    def __init__(self, config):
        super(TrainOT, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = int(config["model"]["input_dim"])
        self.gcn_dim = int(config["model"]["gcn_dim"])
        self.rel_dim = int(config["model"]["rel_dim"])
        self.train_dim = int(config["model"]["train_dim"])
        self.output_dim = int(config["model"]["output_dim"])
        self.num_heads = int(config["model"]["num_heads"])

        self.batches = 1
        self.num_nodes = int(config["dataset"]["nodes"])

        # GCN Layers
        self.gcn1 = GCNConv(self.input_dim, self.gcn_dim)
        self.gcn2 = GCNConv(self.gcn_dim, self.gcn_dim)

        # Train
        self.trainOT = nn.Parameter(torch.randn(self.train_dim))

        # Query, Key, Value, Speak
        self.query = nn.Linear(self.gcn_dim, self.rel_dim)
        self.key = nn.Linear(self.train_dim, self.rel_dim)
        self.value = nn.Linear(self.gcn_dim, self.train_dim)
        self.speak = nn.Linear(self.train_dim, self.output_dim*self.num_nodes)

    def forward(self, batch, timestep):
        # if trainOT is None:
        #     trainOT = torch.randn(self.train_dim, device=self.device)

        # (Timestep, Nodes*Batch, 2) -> (Nodes*Batch, 2)
        # (Timestep, Nodes*Batch, Nodes*Batch) -> (Nodes*Batch, Nodes*Batch)
        # (Batch, Timestep, Node Amt) -> (Time, Batch*Nodes) -> (Batch*Nodes)
        features = batch["big_batch_positions"][timestep]         
        adjacency = batch["big_batched_adjacency_pruned"][timestep]        
        ego_mask = batch['ego_mask_batch'].permute(1, 0, 2)[timestep].reshape(-1)

        features_masked = features[ego_mask]
        adjacency_masked = adjacency[ego_mask][:, ego_mask]
        edge_index_masked = tensor_to_edge_index(adjacency_masked)

        # GCN pass get embed for each node
        gcn1_out = self.gcn1(features_masked, edge_index_masked)
        relu1_out = torch.relu(gcn1_out)
        gcn2_out = self.gcn2(relu1_out, edge_index_masked)

        # Pass embed through query & value
        query_out = self.query(gcn2_out) # (Node, rel_dim)
        value_out = self.value(gcn2_out) # (Node, train_dm)

        # Pass train through key
        key_out = self.key(self.trainOT)

        # Dot prod get relevancy (Node)
        rel = torch.sigmoid((query_out @ key_out) / (self.rel_dim ** 0.5))

        # multiply value * relevancy
        # (Node) * (Node, train_dim) -> (Node, train_dim)
        weighted_rel =  rel.unsqueeze(1) * value_out

        # Add to Train
        summed_idea = torch.sum(weighted_rel, dim=0)
        new_trainOT = self.trainOT + summed_idea

        # Pass Train though speak
        speak_out = self.speak(new_trainOT)

        # Output is X-dim embedding on each node
        # (Batch, Node, Embedding)
        out = speak_out.view(self.batches, -1, self.output_dim)
        return out, new_trainOT

class LSTMOnly(nn.Module):
    """
    Baseline that models only temporal dynamics via LSTM on node features.
    Input: batch dict with 'positions' and 'ego_mask_batch'.
    Output: tensor of shape [B, N, T, output_dim], with inactive nodes zeroed.
    """
    def __init__(self, config):
        super().__init__()
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_timesteps= int(config["dataset"]["timesteps"])
        self.num_nodes    = int(config["dataset"]["nodes"])
        self.batch_size   = int(config["training"]["batch_size"])

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.fc   = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, eval=False):
        # positions: [B, T, N, D], ego_mask: [B, T, N]
        positions = batch['positions'][..., :self.input_dim].to(self.device)
        ego_mask  = batch['ego_mask_batch'].to(self.device)             # [B, T, N]
        B, T, N, D = positions.shape

        # reshape to [B*N, T, D] for LSTM
        feats = positions.permute(0,2,1,3).reshape(B * N, T, D)

        # run LSTM + output projection
        lstm_out, _ = self.lstm(feats)                                  # [B*N, T, hidden_dim]
        out = self.fc(lstm_out)                                         # [B*N, T, output_dim]
        out = out.view(B, N, T, self.output_dim)                        # [B, N, T, output_dim]

        # now mask out inactive nodes/timesteps
        # ego_mask: [B, T, N] -> [B, N, T, 1]
        mask = ego_mask.permute(0,2,1).unsqueeze(-1).type_as(out)
        out = out * mask                                                # zero where mask==0

        return out



class GCNOnly(nn.Module):
    """
    Baseline that applies a static GCN at each time slice independently.
    Input: batch dict with 'big_batch_positions', 'big_batched_adjacency_pruned', 'ego_mask_batch'.
    Output: tensor of shape [B, N, T, output_dim].
    """
    def __init__(self, config):
        super().__init__()
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim   = int(config["model"]["input_dim"])
        self.hidden_dim  = int(config["model"]["hidden_dim"])
        self.output_dim  = int(config["model"]["output_dim"])
        self.num_timesteps = int(config["dataset"]["timesteps"])
        self.num_nodes   = int(config["dataset"]["nodes"])
        self.batch_size  = int(config["training"]["batch_size"])

        self.gcn1 = GCNConv(self.input_dim, self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.fc   = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, eval=False):
        x  = batch['big_batch_positions'].to(self.device)       # [T, B*N, D]
        A  = batch['big_batched_adjacency_pruned'].to(self.device)  # [T, B*N, B*N]
        mask = batch['ego_mask_batch'].permute(1,0,2).reshape(self.num_timesteps, -1).to(self.device)
        B = batch['ego_mask_batch'].shape[0]
        N = self.num_nodes

        outs = []
        for t in range(self.num_timesteps):
            feats_t = x[t]                                  # [B*N, D]
            m_t     = mask[t]                               # [B*N]
            # Mask features and adjacency
            idx     = m_t.nonzero(as_tuple=False).squeeze()
            feats_m = feats_t[idx]
            A_m     = A[t][idx][:, idx]
            edges   = tensor_to_edge_index(A_m)

            h1 = torch.relu(self.gcn1(feats_m, edges))
            h2 = torch.relu(self.gcn2(h1, edges))
            out_feats = self.fc(h2)                         # [num_active, output_dim]

            # Scatter back into placeholder
            placeholder = torch.zeros(B * N, self.output_dim, device=self.device)
            placeholder[idx] = out_feats
            outs.append(placeholder.unsqueeze(0))          # [1, B*N, out_dim]

        h_stack = torch.cat(outs, dim=0)                   # [T, B*N, out_dim]
        h_stack = h_stack.view(self.num_timesteps, B, N, self.output_dim)
        return h_stack.permute(1,2,0,3)                    # [B, N, T, out_dim]


class DynamicGraphNN(nn.Module):
    """
    Baseline that evolves node embeddings over time via GCN + GRUCell.
    Input: same batch dict as for GCNOnly.
    Output: tensor of shape [B, N, T, output_dim].
    """
    def __init__(self, config):
        super().__init__()
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim   = int(config["model"]["input_dim"])
        self.hidden_dim  = int(config["model"]["hidden_dim"])
        self.output_dim  = int(config["model"]["output_dim"])
        self.num_timesteps = int(config["dataset"]["timesteps"])
        self.num_nodes   = int(config["dataset"]["nodes"])
        self.batch_size  = int(config["training"]["batch_size"])

        self.gcn      = GCNConv(self.input_dim, self.hidden_dim)
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc       = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, eval=False):
        x    = batch['big_batch_positions'].to(self.device)    # [T, B*N, D]
        A    = batch['big_batched_adjacency_pruned'].to(self.device)
        mask = batch['ego_mask_batch'].permute(1,0,2).reshape(self.num_timesteps, -1).to(self.device)
        B = batch['ego_mask_batch'].shape[0]
        N = self.num_nodes

        # Initialize hidden state for all nodes
        h_t = torch.zeros(B * N, self.hidden_dim, device=self.device)
        seq = []

        for t in range(self.num_timesteps):
            feats_t = x[t]                                # [B*N, D]
            m_t     = mask[t].nonzero(as_tuple=False).squeeze()
            fm      = feats_t[m_t]
            Am      = A[t][m_t][:, m_t]
            edges   = tensor_to_edge_index(Am)

            gcn_out   = torch.relu(self.gcn(fm, edges))  # [num_active, hidden_dim]
            h_prev    = h_t[m_t]                         # [num_active, hidden_dim]
            h_new     = self.gru_cell(gcn_out, h_prev)   # [num_active, hidden_dim]
            h_t[m_t]  = h_new                            # update only active nodes
            seq.append(h_t.unsqueeze(0))                 # [1, B*N, hidden_dim]

        seq_stack = torch.cat(seq, dim=0)              # [T, B*N, hidden_dim]
        seq_stack = seq_stack.view(self.num_timesteps, B, N, self.hidden_dim)
        out = self.fc(seq_stack)                       # [T, B, N, output_dim]
        return out.permute(1,2,0,3)                    # [B, N, T, output_dim]


class TemporalGCN(nn.Module):
    def __init__(self, config):
        super(TemporalGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"IS CUDA AVALAIBLE: {torch.cuda.is_available()}")
        self.num_nodes = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim = int(config["model"]["input_dim"])
        self.hidden_dim = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim = int(config["model"]["output_dim"])
        self.num_heads = int(config["model"]["num_heads"])

        self.batches = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes
        
        # Graph convolution layers
        self.gcn1 = GCNConv(self.input_dim, self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)
        # Fully connected output layer
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.output_dim)
        
    def forward(self, batch, eval=False):
        ego_mask = batch['ego_mask_batch'] # Shape: (Batch, Timestep, Node Amt)
        x = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batched_adjacency_pruned']

        num_timesteps = x.shape[0]
        num_nodes = x.shape[1]

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

        # Rearrange the placeholder for LSTM processing.
        # Currently: [T, B * max_nodes, hidden_dim] -> reshape to (B*num_nodes, T, hidden_dim)
        x_placeholder = x_placeholder.transpose(0, 1)  # Now shape: (B * max_nodes, T, hidden_dim)


        # LSTM -------
        lstm_out, (h_n, _) = self.lstm(x_placeholder)

        # Reshape to [B, max_nodes, T, hidden_dim]
        embeddings = lstm_out.view(B, max_nodes, num_timesteps, self.hidden_dim)

        embeddings = torch.relu(self.fc1(embeddings))
        embeddings = self.fc2(embeddings)  # [B, max_nodes, T, output_dim]

        return embeddings
        # LSTM -------
        
        # attn_scores = self.attention(lstm_out)  # (B*num_nodes, T, 1)
        # attn_scores = attn_scores.squeeze(-1)   # (B*num_nodes, T)


        # # Compute attention weights.
        # attn_weights = F.softmax(attn_scores, dim=1)  # (B*num_nodes, T)

        # # Compute the context vector as a weighted sum of LSTM outputs (over the time dimension).
        # # Expand attn_weights and multiply elementwise with lstm_out.
        # attn_weights = attn_weights.unsqueeze(-1)    # (B*num_nodes, T, 1)
        # x_out = torch.sum(attn_weights * lstm_out, dim=1)  # (B*num_nodes, hidden_dim)
        
        # # Reshape the context vector to (B, max_nodes, hidden_dim)
        # x_out = x_out.view(B, max_nodes, self.output_dim)


        # return x_out
    
        
class AttentionGCNOld(nn.Module):
    def __init__(self, config):
        super(AttentionGCNOld, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"IS CUDA AVALAIBLE: {torch.cuda.is_available()}")
        self.num_nodes = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim = int(config["model"]["input_dim"])
        self.hidden_dim = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim = int(config["model"]["output_dim"])
        self.num_heads = int(config["model"]["num_heads"])

        self.batches = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        self.register_buffer("pos_embed", self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim))
        
        # Graph convolution layers
        self.gcn1 = GCNConv(self.input_dim, self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)

        self.multi_attention = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
            for _ in range(self.max_nodes)
        ])
        self.query = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.max_nodes)
        ])
        self.value = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.max_nodes)
        ])
        self.key = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.max_nodes)
        ])
        # Fully connected output layer
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.output_dim)

    def get_sinusoidal_encoding(self, timesteps, dim):
        position = torch.arange(timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding
        
    def forward(self, batch, eval=False):
        ego_mask = batch['ego_mask_batch'] # Shape: (Batch, Timestep, Node Amt)
        x = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batched_adjacency_pruned']

        num_timesteps = x.shape[0]
        num_nodes = x.shape[1]

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

        # Rearrange the placeholder for LSTM processing.
        # Currently: [T, B * max_nodes, hidden_dim] -> reshape to (B*num_nodes, T, hidden_dim)
        x_placeholder = x_placeholder.transpose(0, 1)  # Now shape: (B * max_nodes, T, hidden_dim)

        x_placeholder = x_placeholder + self.pos_embed.unsqueeze(0)  # add positional info


        # Each node embedding gets mul by matrix to get query and key vector

        # At each timestep, the node query is dot product by every other node key
        # -> divide by sqrt of that dimension in k/q space
        # -> set forward entries to neg infinity  
        # -> softmaxed

        # ATTENTION ------
        outputs = []
        for attn, query_l, value_l, key_l, node_emb in zip(self.multi_attention, self.query, self.value, self.key, x_placeholder):
            # Add batch dimension
            node_emb = node_emb.unsqueeze(0)  # Now shape: (1, T, hidden_dim)
            # Compute Q, K, V representations
            query = query_l(node_emb)
            key   = key_l(node_emb)
            value = value_l(node_emb)
            # Apply multi-head attention; attn_out will be (1, T, hidden_dim)
            attn_out, _ = attn(query, key, value)
            # Remove the extra batch dimension and collect the result
            outputs.append(attn_out.squeeze(0))
        # Combine all outputs: resulting shape will be (B*max_nodes, T, hidden_dim)
        x_attn = torch.stack(outputs, dim=0)

        x_out = x_attn.view(B, max_nodes, self.num_timesteps, self.hidden_dim)
        
        return x_out
        # ATTENTION ------

        # LSTM -------
        # lstm_out, (h_n, _) = self.lstm(x_placeholder)

        # # Reshape to [B, max_nodes, T, hidden_dim]
        # embeddings = lstm_out.view(B, max_nodes, num_timesteps, self.hidden_dim)

        # embeddings = torch.relu(self.fc1(embeddings))
        # embeddings = self.fc2(embeddings)  # [B, max_nodes, T, output_dim]

        # return embeddings
        # LSTM -------
        
        # attn_scores = self.attention(lstm_out)  # (B*num_nodes, T, 1)
        # attn_scores = attn_scores.squeeze(-1)   # (B*num_nodes, T)


        # # Compute attention weights.
        # attn_weights = F.softmax(attn_scores, dim=1)  # (B*num_nodes, T)

        # # Compute the context vector as a weighted sum of LSTM outputs (over the time dimension).
        # # Expand attn_weights and multiply elementwise with lstm_out.
        # attn_weights = attn_weights.unsqueeze(-1)    # (B*num_nodes, T, 1)
        # x_out = torch.sum(attn_weights * lstm_out, dim=1)  # (B*num_nodes, hidden_dim)
        
        # # Reshape the context vector to (B, max_nodes, hidden_dim)
        # x_out = x_out.view(B, max_nodes, self.output_dim)


        # return x_out

class AttentionGCN(nn.Module):
    def __init__(self, config):
        super(AttentionGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim   = int(config["model"]["input_dim"])
        self.hidden_dim  = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim  = int(config["model"]["output_dim"])
        self.num_heads   = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # sinusoidal positional embedding for the T dimension:
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── Graph convolution layers ───────────────────────────────────────────
        self.gcn1 = GCNConv(self.input_dim,  self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)

        # ─── LSTM (if you still want to use it; can be removed if not needed) ───
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        # ─── replace ModuleList by a single MultiheadAttention ────────────────
        self.multi_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # ─── single set of linear projections (shared across all nodes) ────────
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key   = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        # ─── Fully connected output layer (same as before) ─────────────────────
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)


    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T x dim) sinusoidal positional encoding
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]


    def forward(self, batch, eval=False):
        """
        batch must contain:
          - batch['ego_mask_batch']           : BoolTensor  [B, T, NodeCount]
          - batch['big_batch_positions']      : FloatTensor [T, B*NodeCount, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*NodeCount, B*NodeCount]

        We will return x_out of shape [B, NodeCount, T, hidden_dim] (just as a demo;
        you can plug in fc1/fc2 at the end for actual classification).
        """
        ego_mask             = batch['ego_mask_batch']               # [B, T, N]
        x_raw                = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency  = batch['big_batched_adjacency_pruned']  # [T, B*N, B*N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Build a placeholder to collect per‐timestep GCN outputs ─────────
        # We will fill x_placeholder[t, idx] = GCN‐output for those active nodes idx at time t.
        #
        # Shape = [T, B*N, H], zeros by default
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)

        # Flatten ego_mask → [T, B*N]
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t       = x_raw[t]           # [B*N, input_dim]
            A_t       = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]   # [B*N] bool

            if not mask_flat.any():
                # no nodes active at time t
                continue

            idx      = mask_flat.nonzero(as_tuple=True)[0]  # indices of active rows
            x_t_m    = x_t[idx]                           # [n_active, input_dim]
            A_t_m    = A_t[idx][:, idx]                   # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, num_edges]

            # two GCN layers + ReLU
            h1 = torch.relu(self.gcn1(x_t_m, edge_idx))   # [n_active, H]
            h2 = torch.relu(self.gcn2(h1,    edge_idx))   # [n_active, H]

            # put them back into the big placeholder
            x_placeholder[t, idx, :] = h2

        # ─── Now x_placeholder = [T, B*N, H].  We want to feed (batch = B*N) into attention.
        # Permute → [B*N, T, H] so that MultiheadAttention (with batch_first=True) sees:
        #   batch_size = B*N, seq_len = T, embed_dim = H.
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # Add positional embedding along the T‐axis:
        #   pos_embed: [T, H] → unsqueeze(0) → [1, T, H], broadcast to [B*N, T, H]
        x_seq = x_seq + self.pos_embed.unsqueeze(0)  # [B*N, T, H]

        # Optionally, you could run LSTM here if you still wanted a temporal LSTM step:
        # x_seq, _ = self.lstm(x_seq)  # remains [B*N, T, H]

        # ─── Shared single MultiheadAttention ───────────────────────────────────
        # Project to Q, K, V using the same linear layers for all nodes:
        Q = self.query(x_seq)  # [B*N, T, H]
        K = self.key(x_seq)    # [B*N, T, H]
        V = self.value(x_seq)  # [B*N, T, H]

        # Now run the one shared attention layer:
        #   attn_out: [B*N, T, H], attn_weights: [B*N, T, T] (if you need them)
        attn_out, _ = self.multi_attention(Q, K, V)
        # attn_out[i] is the attended‐over‐time embedding sequence for the i-th node‐in‐batch.

        # ─── Reshape back to [B, N, T, H] so you can do per‐node, per‐time FC or return as-is ──
        x_out = attn_out.view(B, N, T, H)  # [B, N, T, H]

        # If you want to push each (node, time) through the FC layers:
        # flat = x_out.reshape(B * N * T, H)               # [B*N*T, H]
        # h_fc1 = torch.relu(self.fc1(flat))               # [B*N*T, H2]
        # logits = self.fc2(h_fc1)                          # [B*N*T, output_dim]
        # x_out = logits.view(B, N, T, self.output_dim)     # [B, N, T, output_dim]

        return x_out

class DeepAttentionGCN(nn.Module):
    """
    A deeper AttentionGCN that uses:
      - 4 stacked GCNConv layers per timestep instead of 2,
      - a small TransformerEncoder (3 layers) over the time dimension,
      - a 2‐hidden‐layer MLP head at the end.
    """
    def __init__(self, config):
        super(DeepAttentionGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params:
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Sinusoidal positional encoding (T × hidden_dim)
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── FOUR stacked GCNConv layers ────────────────────────────────────────
        self.gcn1 = GCNConv(self.input_dim,    self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn3 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn4 = GCNConv(self.hidden_dim,   self.hidden_dim)

        # ─── A 3‐layer TransformerEncoder over the time dimension ──────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=3)

        # ─── Deep MLP head: hidden_dim → hidden_dim_2 → hidden_dim_2 → output_dim ─
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)


    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T × dim) sinusoidal positional encoding.
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]


    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']           : BoolTensor  [B, T, N]
          - batch['big_batch_positions']      : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, output_dim].
        """
        ego_mask            = batch['ego_mask_batch']               # [B, T, N]
        x_raw               = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned']  # [T, B*N, B*N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Step 1: Run 4 GCNConv layers at each timestep and collect in a placeholder ─────
        # placeholder[t, i] will hold the H‐dim embedding for node i (in the flattened B*N) at time t:
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t      = x_raw[t]            # [B*N, input_dim]
            A_t      = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]    # [B*N] bool

            if not mask_flat.any():
                continue

            idx     = mask_flat.nonzero(as_tuple=True)[0]  # active node indices
            x_t_m   = x_t[idx]                            # [n_active, input_dim]
            A_t_m   = A_t[idx][:, idx]                    # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, E]

            # Four GCN layers, each followed by ReLU
            h1 = F.relu(self.gcn1(x_t_m,  edge_idx))      # [n_active, H]
            h2 = F.relu(self.gcn2(h1,     edge_idx))      # [n_active, H]
            h3 = F.relu(self.gcn3(h2,     edge_idx))      # [n_active, H]
            h4 = F.relu(self.gcn4(h3,     edge_idx))      # [n_active, H]

            # Place h4 back into the placeholder
            x_placeholder[t, idx, :] = h4

        # ─── Step 2: Permute to feed into TransformerEncoder ───────────────────────────
        #    x_placeholder: [T, B*N, H] → permute → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [ (B*N), T, H ]

        # Add positional encoding along the T‐axis:
        #   pos_embed: [T, H] → unsqueeze(0) → [1, T, H], broadcast to [B*N, T, H]
        x_seq = x_seq + self.pos_embed.unsqueeze(0)

        # ─── Step 3: Run a 3‐layer TransformerEncoder over the time dimension ──────────
        # The TransformerEncoderLayer already does: LayerNorm → MHA → Add → Feedforward → Add → LayerNorm
        # Output shape stays [B*N, T, H].
        x_transformed = self.transformer_encoder(x_seq)  # [B*N, T, H]

        # ─── Step 4: Reshape back to [B, N, T, H] so we can do per‐(node,time) classification ─
        x_btnd = x_transformed.view(B, N, T, H)  # [B, N, T, H]

        # ─── Step 5: Run a deeper MLP on each (b,i,t)-embedding → output_dim ─────────────
        # Merge (B, N, T) → single dimension to feed MLP in batch:
        flat = x_btnd.view(B * N * T, H)              # [B*N*T, H]
        h1   = F.relu(self.fc1(flat))                 # [B*N*T, H2]
        h2   = F.relu(self.fc2(h1))                   # [B*N*T, H2]
        logits = self.fc3(h2)                         # [B*N*T, output_dim]
        out = logits.view(B, N, T, self.output_dim)   # [B, N, T, output_dim]

        return out
    
class VeryDeepAttentionGCN(nn.Module):
    """
    A very deep AttentionGCN:
      - Six GCNConv layers per timestep (more spatial depth).
      - A 5‐layer TransformerEncoder over time (more temporal depth).
      - A 4‐hidden‐layer MLP head before final output (more classification depth).
    """
    def __init__(self, config):
        super(VeryDeepAttentionGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Sinusoidal positional encoding (T × hidden_dim)
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── SIX stacked GCNConv layers ──────────────────────────────────────────
        self.gcn1 = GCNConv(self.input_dim,    self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn3 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn4 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn5 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn6 = GCNConv(self.hidden_dim,   self.hidden_dim)

        # ─── A 5‐layer TransformerEncoder over the time dimension ───────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=5)

        # ─── Very deep MLP head: 
        #     hidden_dim → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → output_dim
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc4 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc5 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)


    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T × dim) sinusoidal positional encoding.
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]


    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']           : BoolTensor  [B, T, N]
          - batch['big_batch_positions']      : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, output_dim].
        """
        ego_mask            = batch['ego_mask_batch']               # [B, T, N]
        x_raw               = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned']  # [T, B*N, B*N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Step 1: Run six GCNConv layers at each timestamp and collect into placeholder ─
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t      = x_raw[t]            # [B*N, input_dim]
            A_t      = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]    # [B*N] bool

            if not mask_flat.any():
                continue

            idx      = mask_flat.nonzero(as_tuple=True)[0]  # active indices
            x_t_m    = x_t[idx]                           # [n_active, input_dim]
            A_t_m    = A_t[idx][:, idx]                   # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, E]

            # Six GCN layers, each with ReLU
            h1 = F.relu(self.gcn1(x_t_m,  edge_idx))      # [n_active, H]
            h2 = F.relu(self.gcn2(h1,     edge_idx))      # [n_active, H]
            h3 = F.relu(self.gcn3(h2,     edge_idx))      # [n_active, H]
            h4 = F.relu(self.gcn4(h3,     edge_idx))      # [n_active, H]
            h5 = F.relu(self.gcn5(h4,     edge_idx))      # [n_active, H]
            h6 = F.relu(self.gcn6(h5,     edge_idx))      # [n_active, H]

            x_placeholder[t, idx, :] = h6

        # ─── Step 2: Permute to feed into TransformerEncoder ────────────────────────────
        #    x_placeholder: [T, B*N, H] → permute → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # Add positional encoding along the T‐axis:
        x_seq = x_seq + self.pos_embed.unsqueeze(0)          # [B*N, T, H]

        # ─── Step 3: Run a 5‐layer TransformerEncoder over the time dimension ───────────
        x_transformed = self.transformer_encoder(x_seq)      # [B*N, T, H]

        # ─── Step 4: Reshape back to [B, N, T, H] for per‐(node,time) classification ───
        x_btnd = x_transformed.view(B, N, T, H)              # [B, N, T, H]

        # ─── Step 5: Very deep MLP head: (B*N*T, H) → (B*N*T, output_dim) ─────────────
        flat = x_btnd.view(B * N * T, H)                     # [B*N*T, H]
        h1   = F.relu(self.fc1(flat))                        # [B*N*T, H2]
        h2   = F.relu(self.fc2(h1))                          # [B*N*T, H2]
        h3   = F.relu(self.fc3(h2))                          # [B*N*T, H2]
        h4   = F.relu(self.fc4(h3))                          # [B*N*T, H2]
        logits = self.fc5(h4)                                 # [B*N*T, output_dim]
        out = logits.view(B, N, T, self.output_dim)           # [B, N, T, output_dim]

        return out


class VeryDeepAttentionDropOutGCN(nn.Module):
    """
    A very deep AttentionGCN with dropout:
      - Six GCNConv layers per timestep (spatial depth), each followed by ReLU + Dropout.
      - A 5‐layer TransformerEncoder over time (it already includes its own dropout).
      - A 4‐hidden‐layer MLP head, with Dropout between each hidden layer.
    """
    def __init__(self, config):
        super(VeryDeepAttentionDropOutGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Dropout probability (you can also pull this from config)
        self.dropout_prob = 0.5
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Sinusoidal positional encoding (T × hidden_dim)
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── SIX stacked GCNConv layers ──────────────────────────────────────────
        # Each GCNConv will be followed by ReLU + Dropout in forward.
        self.gcn1 = GCNConv(self.input_dim,    self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn3 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn4 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn5 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn6 = GCNConv(self.hidden_dim,   self.hidden_dim)

        # ─── A 5‐layer TransformerEncoder over the time dimension ───────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,            # transformer’s internal dropout
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=5)

        # ─── Very deep MLP head:
        #     hidden_dim → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → output_dim
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc4 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc5 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)

    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T × dim) sinusoidal positional encoding.
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]

    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']           : BoolTensor  [B, T, N]
          - batch['big_batch_positions']      : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, output_dim].
        """
        ego_mask            = batch['ego_mask_batch']               # [B, T, N]
        x_raw               = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned'] # [T, B*N, B*N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Step 1: Run six GCNConv layers at each timestamp and collect into placeholder ─
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t      = x_raw[t]            # [B*N, input_dim]
            A_t      = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]    # [B*N] bool

            if not mask_flat.any():
                continue

            idx      = mask_flat.nonzero(as_tuple=True)[0]  # active indices
            x_t_m    = x_t[idx]                           # [n_active, input_dim]
            A_t_m    = A_t[idx][:, idx]                   # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, E]

            # Six GCN layers, each with ReLU + Dropout
            h1 = F.relu(self.gcn1(x_t_m,  edge_idx))   # [n_active, H]
            h1 = self.dropout(h1)

            h2 = F.relu(self.gcn2(h1,     edge_idx))   # [n_active, H]
            h2 = self.dropout(h2)

            h3 = F.relu(self.gcn3(h2,     edge_idx))   # [n_active, H]
            h3 = self.dropout(h3)

            h4 = F.relu(self.gcn4(h3,     edge_idx))   # [n_active, H]
            h4 = self.dropout(h4)

            h5 = F.relu(self.gcn5(h4,     edge_idx))   # [n_active, H]
            h5 = self.dropout(h5)

            h6 = F.relu(self.gcn6(h5,     edge_idx))   # [n_active, H]
            h6 = self.dropout(h6)

            x_placeholder[t, idx, :] = h6

        # ─── Step 2: Permute to feed into TransformerEncoder ────────────────────────────
        #    x_placeholder: [T, B*N, H] → permute → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # Add positional encoding along the T‐axis:
        x_seq = x_seq + self.pos_embed.unsqueeze(0)          # [B*N, T, H]

        # ─── Step 3: Run a 5‐layer TransformerEncoder over the time dimension ───────────
        x_transformed = self.transformer_encoder(x_seq)      # [B*N, T, H]

        # ─── Step 4: Reshape back to [B, N, T, H] for per‐(node,time) classification ───
        x_btnd = x_transformed.view(B, N, T, H)              # [B, N, T, H]

        # ─── Step 5: Very deep MLP head: (B*N*T, H) → (B*N*T, output_dim) ─────────────
        flat = x_btnd.view(B * N * T, H)                     # [B*N*T, H]

        h1   = F.relu(self.fc1(flat))                        # [B*N*T, H2]
        h1   = self.dropout(h1)

        h2   = F.relu(self.fc2(h1))                          # [B*N*T, H2]
        h2   = self.dropout(h2)

        h3   = F.relu(self.fc3(h2))                          # [B*N*T, H2]
        h3   = self.dropout(h3)

        h4   = F.relu(self.fc4(h3))                          # [B*N*T, H2]
        h4   = self.dropout(h4)

        logits = self.fc5(h4)                                # [B*N*T, output_dim]
        out = logits.view(B, N, T, self.output_dim)           # [B, N, T, output_dim]

        return out


class JustAttentionDropOutGCN(nn.Module):
    """
    A very deep AttentionGCN with dropout:
      - Six GCNConv layers per timestep (spatial depth), each followed by ReLU + Dropout.
      - A 5‐layer TransformerEncoder over time (it already includes its own dropout).
      - A 4‐hidden‐layer MLP head, with Dropout between each hidden layer.
    """
    def __init__(self, config):
        super(JustAttentionDropOutGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Dropout probability (you can also pull this from config)
        self.dropout_prob = 0.1
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Sinusoidal positional encoding (T × hidden_dim)
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── SIX stacked GCNConv layers ──────────────────────────────────────────
        # Each GCNConv will be followed by ReLU + Dropout in forward.
        self.gcn1 = GCNConv(self.input_dim,    self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn3 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn4 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn5 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn6 = GCNConv(self.hidden_dim,   self.hidden_dim)

        # ─── A 5‐layer TransformerEncoder over the time dimension ───────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,            # transformer’s internal dropout
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=5)

        # ─── Very deep MLP head:
        #     hidden_dim → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → output_dim
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc4 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc5 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)

    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T × dim) sinusoidal positional encoding.
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]

    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']           : BoolTensor  [B, T, N]
          - batch['big_batch_positions']      : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, output_dim].
        """
        ego_mask            = batch['ego_mask_batch']               # [B, T, N]
        x_raw               = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned'] # [T, B*N, B*N]

        T = self.num_timesteps
        O = self.output_dim
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Step 1: Run six GCNConv layers at each timestamp and collect into placeholder ─
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t      = x_raw[t]            # [B*N, input_dim]
            A_t      = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]    # [B*N] bool

            if not mask_flat.any():
                continue

            idx      = mask_flat.nonzero(as_tuple=True)[0]  # active indices
            x_t_m    = x_t[idx]                           # [n_active, input_dim]
            A_t_m    = A_t[idx][:, idx]                   # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, E]

            # Six GCN layers, each with ReLU + Dropout
            h1 = F.relu(self.gcn1(x_t_m,  edge_idx))   # [n_active, H]
            h1 = self.dropout(h1)

            h2 = F.relu(self.gcn2(h1,     edge_idx))   # [n_active, H]
            h2 = self.dropout(h2)

            h3 = F.relu(self.gcn3(h2,     edge_idx))   # [n_active, H]
            h3 = self.dropout(h3)

            h4 = F.relu(self.gcn4(h3,     edge_idx))   # [n_active, H]
            h4 = self.dropout(h4)

            h5 = F.relu(self.gcn5(h4,     edge_idx))   # [n_active, H]
            h5 = self.dropout(h5)

            h6 = F.relu(self.gcn6(h5,     edge_idx))   # [n_active, H]
            h6 = self.dropout(h6)

            x_placeholder[t, idx, :] = h6

        # ─── Step 2: Permute to feed into TransformerEncoder ────────────────────────────
        #    x_placeholder: [T, B*N, H] → permute → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # Add positional encoding along the T‐axis:
        x_seq = x_seq + self.pos_embed.unsqueeze(0)          # [B*N, T, H]

        # ─── Step 3: Run a 5‐layer TransformerEncoder over the time dimension ───────────
        x_transformed = self.transformer_encoder(x_seq)      # [B*N, T, H]

        # ─── Step 4: Reshape back to [B, N, T, H] for per‐(node,time) classification ───
        out = x_transformed.view(B, N, T, H)              # [B, N, T, H]
        # out = x_transformed.view(B, N, T, O)              # [B, N, T, O]

        return out



class JustAttention2GCN(nn.Module):
    """
    A very deep AttentionGCN with dropout, now including:
      - Residual connections between GCN layers
      - LayerNorm after each GCNConv  (replacing BatchNorm1d)
      - Learnable positional embeddings
      - A padding mask for the TransformerEncoder
      - Six GCNConv layers per timestep (spatial depth), each followed by LayerNorm, ReLU + Dropout
      - A 5‐layer TransformerEncoder over time (with padding mask)
      - A 4‐hidden‐layer MLP head
    """
    def __init__(self, config):
        super(JustAttention2GCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Dropout probability
        self.dropout_prob = 0.1
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # ─── Learnable positional embeddings ──────────────────────────────────────
        # shape = [T, H]
        self.pos_embed = nn.Embedding(self.num_timesteps, self.hidden_dim)

        # ─── SIX stacked GCNConv layers ──────────────────────────────────────────
        # Each followed by LayerNorm, ReLU, Dropout, and with residuals where dims match
        self.gcn1 = GCNConv(self.input_dim,  self.hidden_dim)
        self.ln1  = nn.LayerNorm(self.hidden_dim)

        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.ln2  = nn.LayerNorm(self.hidden_dim)

        self.gcn3 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.ln3  = nn.LayerNorm(self.hidden_dim)

        self.gcn4 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.ln4  = nn.LayerNorm(self.hidden_dim)

        self.gcn5 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.ln5  = nn.LayerNorm(self.hidden_dim)

        self.gcn6 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.ln6  = nn.LayerNorm(self.hidden_dim)

        # ─── A 5‐layer TransformerEncoder over the time dimension ────────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_prob,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=5)

        # ─── Very deep MLP head ───────────────────────────────────────────────────
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc4 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc5 = nn.Linear(self.hidden_dim_2, self.output_dim)

        # Move everything to the correct device
        self.to(self.device)

    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']             : BoolTensor  [B, T, N]
          - batch['big_batch_positions']        : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, output_dim].
        """

        # ─── 1) Move all batch inputs to self.device ───────────────────────────────
        ego_mask            = batch['ego_mask_batch'].to(self.device)               # [B, T, N]
        x_raw               = batch['big_batch_positions'].to(self.device)          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned'].to(self.device) # [T, B*N, B*N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── 2) Build placeholders and flatten masks ────────────────────────────────
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        # ─── 3) Run six GCNConv layers at each timestep, with LayerNorm & residuals ─
        for t in range(T):
            x_t      = x_raw[t]               # [B*N, input_dim]
            A_t      = big_batch_adjacency[t] # [B*N, B*N]
            mask_flat = ego_mask_flat[t]      # [B*N] bool

            if not mask_flat.any():
                continue

            idx   = mask_flat.nonzero(as_tuple=True)[0]  # active indices
            x_t_m = x_t[idx]                             # [n_active, input_dim]
            A_t_m = A_t[idx][:, idx]                     # [n_active, n_active]

            # ─── 3a) Convert to edge_index (on CPU), then move to self.device ─────────
            edge_idx = tensor_to_edge_index(A_t_m)       # [2, E] on CPU
            edge_idx = edge_idx.to(self.device)          # now on GPU if needed

            # ─── 3b) Six GCN layers with LayerNorm, ReLU + Dropout & residuals ───────
            # Layer 1 (no residual from input since dims differ)
            h1 = self.gcn1(x_t_m, edge_idx)        # → [n_active, H]
            h1 = self.ln1(h1)
            h1 = F.relu(h1)
            h1 = self.dropout(h1)

            # Layer 2 (residual from h1)
            h2_raw = self.gcn2(h1, edge_idx)       # → [n_active, H]
            h2 = self.ln2(h2_raw)
            h2 = F.relu(h2 + h1)
            h2 = self.dropout(h2)

            # Layer 3 (residual from h2)
            h3_raw = self.gcn3(h2, edge_idx)       # → [n_active, H]
            h3 = self.ln3(h3_raw)
            h3 = F.relu(h3 + h2)
            h3 = self.dropout(h3)

            # Layer 4 (residual from h3)
            h4_raw = self.gcn4(h3, edge_idx)       # → [n_active, H]
            h4 = self.ln4(h4_raw)
            h4 = F.relu(h4 + h3)
            h4 = self.dropout(h4)

            # Layer 5 (residual from h4)
            h5_raw = self.gcn5(h4, edge_idx)       # → [n_active, H]
            h5 = self.ln5(h5_raw)
            h5 = F.relu(h5 + h4)
            h5 = self.dropout(h5)

            # Layer 6 (residual from h5)
            h6_raw = self.gcn6(h5, edge_idx)       # → [n_active, H]
            h6 = self.ln6(h6_raw)
            h6 = F.relu(h6 + h5)
            h6 = self.dropout(h6)

            # Write back into placeholder
            x_placeholder[t, idx, :] = h6         # [n_active, H]

        # ─── 4) Permute to feed into TransformerEncoder ─────────────────────────────
        # x_placeholder: [T, B*N, H] → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # ─── 5) Build padding mask for Transformer: True where position should be masked (inactive) ─
        active_mask = ego_mask_flat.permute(1, 0)            # [B*N, T], True=active
        padding_mask = ~active_mask                          # [B*N, T], True=inactive

        # ─── 6) Add learnable positional embeddings ────────────────────────────────
        # Make a [B*N, T] index-tensor: each row = [0,1,2,…,T-1]
        time_indices = torch.arange(T, device=self.device).unsqueeze(0).repeat(B * N, 1)
        pos = self.pos_embed(time_indices)                   # [B*N, T, H]
        x_seq = x_seq + pos                                  # add learnable pos‐embed

        # ─── 7) Run a 5‐layer TransformerEncoder over time with padding_mask ────────
        x_transformed = self.transformer_encoder(
            x_seq,
            src_key_padding_mask=padding_mask  # mask out inactive timesteps per node
        )  # → [B*N, T, H]

        # ─── 8) Reshape back to [B, N, T, H] ───────────────────────────────────────
        out = x_transformed.view(B, N, T, H)  # [B, N, T, H]

        # ─── 9) MLP head (H → hidden_dim_2 → … → output_dim) ───────────────────────
        # h_flat = h.view(B * N * T, H)                     # [B*N*T, H]
        # h_mlp = F.relu(self.dropout(self.fc1(h_flat)))    # [B*N*T, hidden_dim_2]
        # h_mlp = F.relu(self.dropout(self.fc2(h_mlp)))     # [B*N*T, hidden_dim_2]
        # h_mlp = F.relu(self.dropout(self.fc3(h_mlp)))     # [B*N*T, hidden_dim_2]
        # h_mlp = F.relu(self.dropout(self.fc4(h_mlp)))     # [B*N*T, hidden_dim_2]
        # out   = self.fc5(h_mlp)                            # [B*N*T, output_dim]
        # out   = out.view(B, N, T, self.output_dim)         # [B, N, T, output_dim]

        return out

class JustAttentionDropOutGAT(nn.Module):
    """
    A very deep Attention‐GAT with dropout:
      - Six GATConv layers per timestep (spatial depth), each followed by ReLU + Dropout.
      - A 5‐layer TransformerEncoder over time (it already includes its own dropout).
      - A 4‐hidden‐layer MLP head (not shown in forward here), with Dropout between each hidden layer.
    """
    def __init__(self, config):
        super(JustAttentionDropOutGAT, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Dropout probability
        self.dropout_prob = 0.1
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Sinusoidal positional encoding (T × hidden_dim)
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── SIX stacked GATConv layers ───────────────────────────────────────────
        # Each GATConv will be followed by ReLU + Dropout in forward.
        # We set concat=False so that output shape is [n_active, hidden_dim] even with multiple heads.
        self.gat1 = GATConv(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout_prob
        )
        self.gat2 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout_prob
        )
        self.gat3 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout_prob
        )
        self.gat4 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout_prob
        )
        self.gat5 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout_prob
        )
        self.gat6 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout_prob
        )

        # ─── A 5‐layer TransformerEncoder over the time dimension ───────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,            # transformer’s internal dropout
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=5)

        # ─── Very deep MLP head (not used in forward here) ───────────────────────
        self.fc1 = nn.Linear(self.hidden_dim,   self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc4 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc5 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)

    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T × dim) sinusoidal positional encoding.
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]

    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']           : BoolTensor  [B, T, N]
          - batch['big_batch_positions']      : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, hidden_dim].
        """
        ego_mask            = batch['ego_mask_batch']               # [B, T, N]
        x_raw               = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned'] # [T, B*N, B*N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Step 1: Run six GATConv layers at each timestamp and collect into placeholder ─
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t      = x_raw[t]            # [B*N, input_dim]
            A_t      = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]    # [B*N] bool

            if not mask_flat.any():
                continue

            # Select only active nodes at this time:
            idx      = mask_flat.nonzero(as_tuple=True)[0]  # [n_active]
            x_t_m    = x_t[idx]                           # [n_active, input_dim]
            A_t_m    = A_t[idx][:, idx]                   # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, E]

            # Six GAT layers, each with ReLU + Dropout
            h1 = F.relu(self.gat1(x_t_m,  edge_idx))   # [n_active, H]
            h1 = self.dropout(h1)

            h2 = F.relu(self.gat2(h1,       edge_idx))   # [n_active, H]
            h2 = self.dropout(h2)

            h3 = F.relu(self.gat3(h2,       edge_idx))   # [n_active, H]
            h3 = self.dropout(h3)

            h4 = F.relu(self.gat4(h3,       edge_idx))   # [n_active, H]
            h4 = self.dropout(h4)

            h5 = F.relu(self.gat5(h4,       edge_idx))   # [n_active, H]
            h5 = self.dropout(h5)

            h6 = F.relu(self.gat6(h5,       edge_idx))   # [n_active, H]
            h6 = self.dropout(h6)

            x_placeholder[t, idx, :] = h6

        # ─── Step 2: Permute to feed into TransformerEncoder ────────────────────────────
        #    x_placeholder: [T, B*N, H] → permute → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # Add positional encoding along the T‐axis:
        x_seq = x_seq + self.pos_embed.unsqueeze(0)          # [B*N, T, H]

        # ─── Step 3: Run a 5‐layer TransformerEncoder over the time dimension ───────────
        x_transformed = self.transformer_encoder(x_seq)      # [B*N, T, H]

        # ─── Step 4: Reshape back to [B, N, T, H] for per‐(node,time) representation ───
        out = x_transformed.view(B, N, T, H)              # [B, N, T, H]

        return out

class AlmostAttentionGCN(nn.Module):
    """
    A very deep AttentionGCN with dropout:
      - Six GCNConv layers per timestep (spatial depth), each followed by ReLU + Dropout.
      - A 5‐layer TransformerEncoder over time (it already includes its own dropout).
      - A 4‐hidden‐layer MLP head, with Dropout between each hidden layer.
    """
    def __init__(self, config):
        super(AlmostAttentionGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset/model params
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.num_timesteps = int(config["dataset"]["timesteps"])

        self.input_dim    = int(config["model"]["input_dim"])
        self.hidden_dim   = int(config["model"]["hidden_dim"])
        self.hidden_dim_2 = int(config["model"]["hidden_dim_2"])
        self.output_dim   = int(config["model"]["output_dim"])
        self.num_heads    = int(config["model"]["num_heads"])

        self.batches   = int(config["training"]["batch_size"])
        self.max_nodes = self.batches * self.num_nodes

        # Dropout probability (you can also pull this from config)
        self.dropout_prob = 0.1
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Sinusoidal positional encoding (T × hidden_dim)
        self.register_buffer(
            "pos_embed",
            self.get_sinusoidal_encoding(self.num_timesteps, self.hidden_dim),
        )

        # ─── SIX stacked GCNConv layers ──────────────────────────────────────────
        # Each GCNConv will be followed by ReLU + Dropout in forward.
        self.gcn1 = GCNConv(self.input_dim,    self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn3 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn4 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn5 = GCNConv(self.hidden_dim,   self.hidden_dim)
        self.gcn6 = GCNConv(self.hidden_dim,   self.hidden_dim)

        # ─── A 5‐layer TransformerEncoder over the time dimension ───────────────
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,            # transformer’s internal dropout
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=5)

        # ─── Very deep MLP head:
        #     hidden_dim → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → hidden_dim_2 → output_dim
        self.fc1 = nn.Linear(self.hidden_dim,   self.output_dim)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc4 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)
        self.fc5 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self.to(self.device)

    def get_sinusoidal_encoding(self, timesteps, dim):
        """
        Build a (T × dim) sinusoidal positional encoding.
        """
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(timesteps, dim, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding  # shape = [T, dim]

    def forward(self, batch, eval=False):
        """
        Inputs (from `batch`):
          - batch['ego_mask_batch']           : BoolTensor  [B, T, N]
          - batch['big_batch_positions']      : FloatTensor [T, B*N, input_dim]
          - batch['big_batched_adjacency_pruned']: FloatTensor [T, B*N, B*N]

        Returns a tensor of shape [B, N, T, output_dim].
        """
        ego_mask            = batch['ego_mask_batch']               # [B, T, N]
        x_raw               = batch['big_batch_positions']          # [T, B*N, input_dim]
        big_batch_adjacency = batch['big_batched_adjacency_pruned'] # [T, B*N, B*N]

        T = self.num_timesteps
        O = self.output_dim
        B = ego_mask.shape[0]
        N = ego_mask.shape[2]
        H = self.hidden_dim

        # ─── Step 1: Run six GCNConv layers at each timestamp and collect into placeholder ─
        x_placeholder = torch.zeros(T, B * N, H, device=self.device)
        ego_mask_flat = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        for t in range(T):
            x_t      = x_raw[t]            # [B*N, input_dim]
            A_t      = big_batch_adjacency[t]  # [B*N, B*N]
            mask_flat = ego_mask_flat[t]    # [B*N] bool

            if not mask_flat.any():
                continue

            idx      = mask_flat.nonzero(as_tuple=True)[0]  # active indices
            x_t_m    = x_t[idx]                           # [n_active, input_dim]
            A_t_m    = A_t[idx][:, idx]                   # [n_active, n_active]
            edge_idx = tensor_to_edge_index(A_t_m)        # [2, E]

            # Six GCN layers, each with ReLU + Dropout
            h1 = F.relu(self.gcn1(x_t_m,  edge_idx))   # [n_active, H]
            h1 = self.dropout(h1)

            h2 = F.relu(self.gcn2(h1,     edge_idx))   # [n_active, H]
            h2 = self.dropout(h2)

            h3 = F.relu(self.gcn3(h2,     edge_idx))   # [n_active, H]
            h3 = self.dropout(h3)

            h4 = F.relu(self.gcn4(h3,     edge_idx))   # [n_active, H]
            h4 = self.dropout(h4)

            h5 = F.relu(self.gcn5(h4,     edge_idx))   # [n_active, H]
            h5 = self.dropout(h5)

            h6 = F.relu(self.gcn6(h5,     edge_idx))   # [n_active, H]
            h6 = self.dropout(h6)

            x_placeholder[t, idx, :] = h6

        # ─── Step 2: Permute to feed into TransformerEncoder ────────────────────────────
        #    x_placeholder: [T, B*N, H] → permute → [B*N, T, H]
        x_seq = x_placeholder.permute(1, 0, 2).contiguous()  # [B*N, T, H]

        # Add positional encoding along the T‐axis:
        x_seq = x_seq + self.pos_embed.unsqueeze(0)          # [B*N, T, H]

        # ─── Step 3: Run a 5‐layer TransformerEncoder over the time dimension ───────────
        x_transformed = self.transformer_encoder(x_seq)      # [B*N, T, H]

        # ─── Step 4: Reshape back to [B, N, T, H] for per‐(node,time) classification ───
        out = x_transformed.view(B, N, T, H)              # [B, N, T, H]
        # out = x_transformed.view(B, N, T, O)              # [B, N, T, O]

        out = self.fc1(out)              # now out is [B, N, T, O]

        return out
    
class GCNSE(nn.Module):
    """
    GCN‐SE (Squeeze‐and‐Excitation) for dynamic node classification. 
    Inputs:
      - batch['big_batch_positions']:  Tensor of shape [T, B*N, D_in]
      - batch['big_batched_adjacency_pruned']: Tensor of shape [T, B*N, B*N]
      - batch['ego_mask_batch']: Tensor of shape [B, T, N]
    Output:
      - Tensor of shape [B, N, T, output_dim]
        (we place the final weighted node embeddings into every time‐slice;
         downstream training can just take the last slice if desired).
    """
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------------------
        # 1) Read config parameters:
        # -------------------------------------------------------------------
        self.input_dim     = int(config["model"]["input_dim"])
        self.hidden_dim    = int(config["model"]["hidden_dim"])
        self.output_dim    = int(config["model"]["output_dim"])
        self.num_timesteps = int(config["dataset"]["timesteps"])
        self.num_nodes     = int(config["dataset"]["nodes"])
        self.batch_size    = int(config["training"]["batch_size"])

        # -------------------------------------------------------------------
        # 2) GCN‐backbone (two GCNConv layers):
        #    We use the same conv‐sizes for every timestep. After masking
        #    (via ego_mask), each subgraph is passed through gcn1→ReLU→gcn2.
        # -------------------------------------------------------------------
        self.gcn1 = GCNConv(self.input_dim,  self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)

        # -------------------------------------------------------------------
        # 3) “Squeeze‐and‐Excitation” block on the per‐timestep outputs:
        #    We collapse each Z_t^(2) (size [num_active, hidden_dim]) into a
        #    single scalar c_t = mean(Z_t^(2)).  Stacking all T scalars yields
        #    c ∈ ℝ^T.  Then:
        #        s = sigmoid( W2 · ReLU(W1 · c) ),   W1: (T → T//r), W2: (T//r → T)
        #    We choose r=2 (i.e. hidden size = T//2).  The final s ∈ ℝ^T are
        #    our attention weights “Watt”.
        # -------------------------------------------------------------------
        r = 2
        squeeze_dim = self.num_timesteps // r
        self.se_fc1 = nn.Linear(self.num_timesteps, squeeze_dim)
        self.se_fc2 = nn.Linear(squeeze_dim,      self.num_timesteps)

        # -------------------------------------------------------------------
        # 4) Final “speak”‐layer: project the weighted sum of all Z_t^(2) to
        #    output_dim.  In the paper, that final vector (per node) is fed
        #    to a classifier; here we simply return a [B, N, T, output_dim]
        #    tensor so that it matches the other models’ conventions.  Downstream
        #    code can take the last‐slice if it wants only “Θ̂_T”.
        # -------------------------------------------------------------------
        self.out_fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, eval=False):
        """
        batch dict contains:
          - batch['big_batch_positions']       : [T, B*N,  D_in]
          - batch['big_batched_adjacency_pruned']: [T, B*N, B*N]
          - batch['ego_mask_batch']            : [B, T, N]  (Boolean or 0/1)
        Returns:
          - output  : [B, N, T, output_dim]
        """
        # Move everything to device:
        x_all = batch["big_batch_positions"].to(self.device)            # [T, B*N, D_in]
        A_all = batch["big_batched_adjacency_pruned"].to(self.device)   # [T, B*N, B*N]
        ego_mask = batch["ego_mask_batch"].to(self.device)              # [B, T, N]

        T = self.num_timesteps
        B = ego_mask.shape[0]
        N = self.num_nodes
        D = self.input_dim

        # Reformat ego_mask → [T, B*N] mask per timestep:
        #    ego_mask: [B, T, N]  →  [T, B, N] → reshape → [T, B*N]
        mask_t_bn = ego_mask.permute(1, 0, 2).reshape(T, B * N)  # [T, B*N]

        # -------------------------------------------------------------------
        # Step 1: run “first” 2‐layer GCN on each (timestep, masked) subgraph,
        #         gather Z_t^(2) as well as a global mean c_t (≈“squeeze”).
        # We'll store:
        #   placeholder_z[t]  =  size [B*N, hidden_dim], where inactive nodes have zero
        #   c_list[t]         =  single scalar (= mean over all active Z_t^(2))
        # -------------------------------------------------------------------
        placeholder_z = torch.zeros(T, B * N, self.hidden_dim, device=self.device)
        c_list = [None] * T

        for t in range(T):
            mask_flat = mask_t_bn[t]                 # [B*N]  (0/1)
            idx_active = mask_flat.nonzero(as_tuple=True)[0]  # indices of active nodes

            # 1) Mask features & adjacency:
            x_t = x_all[t]              # [B*N, D_in]
            A_t = A_all[t]              # [B*N, B*N]
            if idx_active.numel() == 0:
                # If no active nodes at this timestep, c_t = 0 and continue
                c_list[t] = torch.tensor(0.0, device=self.device)
                continue

            x_t_active = x_t[idx_active]                     # [num_active, D_in]
            A_t_active = A_t[idx_active][:, idx_active]      # [num_active, num_active]
            edge_index = tensor_to_edge_index(A_t_active)    # [2, E]

            # 2) GCN pass:
            h1 = self.gcn1(x_t_active, edge_index)       # [num_active, hidden_dim]
            h1 = F.relu(h1)
            h2 = self.gcn2(h1, edge_index)               # [num_active, hidden_dim]

            # 3) “Squeeze”: global average of h2 over all active nodes and all feature dims:
            #    (We want a single scalar c_t := mean(h2))
            c_list[t] = h2.mean()

            # 4) “Placeholder”: put those h2‐embeddings back into a [B*N, hidden_dim] array,
            #    zero‐padding for inactive nodes:
            placeholder = torch.zeros(B * N, self.hidden_dim, device=self.device)
            placeholder[idx_active] = h2
            placeholder_z[t] = placeholder

        # Stack c_t scalars → c_vec ∈ ℝ^T
        c_vec = torch.stack(c_list, dim=0)            # [T]

        # -------------------------------------------------------------------
        # Step 2: “Excitation” block → compute attention weights s ∈ ℝ^T:
        #           s = sigmoid( se_fc2( ReLU( se_fc1(c_vec) ) ) )
        # -------------------------------------------------------------------
        # NOTE: se_fc1 expects shape [ batch_size, in_features ], but here we just
        # have a single vector c_vec of length T.  We simply unsqueeze(0) so that
        # it’s shape [1, T], run through linear layers, then squeeze back to [T].
        s = self.se_fc1(c_vec.unsqueeze(0))           # → [1, T//r]
        s = F.relu(s)
        s = self.se_fc2(s)                            # [1, T]
        s = torch.sigmoid(s).squeeze(0)               # [T]

        # -------------------------------------------------------------------
        # Step 3: Weighted sum across all T placeholders:
        #    Z_hat = ∑_{t=0..T-1} s[t] * placeholder_z[t]
        #    → Z_hat has shape [B*N, hidden_dim]
        # -------------------------------------------------------------------
        Z_hat = torch.zeros(B * N, self.hidden_dim, device=self.device)
        for t in range(T):
            Z_hat += s[t] * placeholder_z[t]          # [B*N, hidden_dim]

        # -------------------------------------------------------------------
        # Step 4: Project Z_hat → output_dim, reshape to [B, N, output_dim],
        #         then repeat across T so that final shape is [B, N, T, output_dim].
        # -------------------------------------------------------------------
        out_embed = self.out_fc(Z_hat)                # [B*N, output_dim]
        out_embed = out_embed.view(B, N, self.output_dim)  # [B, N, output_dim]

        # “Duplicate” across T so that shape → [B, N, T, output_dim]:
        out_embed = out_embed.unsqueeze(2).repeat(1, 1, T, 1)

        return out_embed
    
    
class TGATLayer(nn.Module):
    def __init__(self, node_dim, time_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.total_dim = node_dim + time_dim

        self.query = nn.Linear(self.total_dim, out_dim)
        self.key   = nn.Linear(self.total_dim, out_dim)
        self.value = nn.Linear(self.total_dim, out_dim)
        self.out   = nn.Linear(out_dim, out_dim)

    def forward(self, node_feats, time_feats):
        xt = torch.cat([node_feats, time_feats], dim=-1)  # [N, D + T]
        Q, K, V = self.query(xt), self.key(xt), self.value(xt)
        scores = Q @ K.T / (self.out_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        agg = attn @ V
        return self.out(agg)

class TGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim  = int(config["model"]["input_dim"])
        self.hidden_dim = int(config["model"]["hidden_dim"])
        self.output_dim = int(config["model"]["output_dim"])
        self.time_dim   = int(config["model"].get("time_dim", 16))
        self.num_timesteps = int(config["dataset"]["timesteps"])
        self.num_nodes  = int(config["dataset"]["nodes"])
        self.batch_size = int(config["training"]["batch_size"])

        self.gcn1 = GCNConv(self.input_dim, self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.time_proj = nn.Linear(1, self.time_dim)
        self.tgat_layer = TGATLayer(self.hidden_dim, self.time_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, eval=False):
        x = batch['big_batch_positions'].to(self.device)
        A = batch['big_batched_adjacency_pruned'].to(self.device)
        mask = batch['ego_mask_batch'].permute(1, 0, 2).reshape(self.num_timesteps, -1).to(self.device)

        B, N = batch['ego_mask_batch'].shape[0], self.num_nodes
        outs = []

        for t in range(self.num_timesteps):
            xt = x[t]
            At = A[t]
            mt = mask[t].nonzero(as_tuple=False).squeeze()
            if mt.numel() == 0:  # skip if no active nodes
                outs.append(torch.zeros(B * N, self.output_dim, device=self.device).unsqueeze(0))
                continue

            xt_masked = xt[mt]
            At_masked = At[mt][:, mt]
            edge_index = torch.nonzero(At_masked, as_tuple=False).T

            h = F.relu(self.gcn1(xt_masked, edge_index))
            h = F.relu(self.gcn2(h, edge_index))

            t_enc = torch.full((h.size(0), 1), float(t), device=self.device)
            t_vec = torch.sin(self.time_proj(t_enc))

            h_tgat = self.tgat_layer(h, t_vec)

            out = self.fc(h_tgat)
            full = torch.zeros(B * N, self.output_dim, device=self.device)
            full[mt] = out
            outs.append(full.unsqueeze(0))

        out_stack = torch.cat(outs, dim=0)  # [T, B*N, output_dim]
        return out_stack.view(self.num_timesteps, B, N, self.output_dim).permute(1, 2, 0, 3)  # [B, N, T, D]
    
if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    dir_path = str(config["dataset"]["dir_path"])
    dataset_name = str(config["dataset"]["dataset_name"])
    val_name="{}{}_val.pt".format(dir_path, dataset_name)

    dataset = GCNDataset(val_name)


    batch_size = int(config["training"]["batch_size"])
    time_steps = int(config["dataset"]["timesteps"])

    # Define the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = TemporalGCN(config).to(device)
    model = TrainOT(config).to(device)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions'] # batch, timestamp, node_amt, 3
        ego_mask_batch = batch['ego_mask_batch']
        big_batch_positions = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batch_adjacency']
        trainOT_state = None
        for time in range(time_steps):
            emb, trainOT_state = model(batch, time, trainOT_state)

            print(emb.shape)

            exit()
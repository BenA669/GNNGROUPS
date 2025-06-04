import torch
import torch.nn as nn
from datasetEpisode import GCNDataset, collate_fn
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
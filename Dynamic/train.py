import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from model import TemporalGCN  # Your model from model.py


class GCNDataset(Dataset):
    """
    A Dataset class that loads the data saved in *.pt files (list of (positions, adjacency)).
    Each item:
        positions: (time_steps, node_amt, 3)  -- [x, y, group_id]
        adjacency: (time_steps, node_amt, node_amt)
    """
    def __init__(self, data_path):
        super().__init__()
        self.data = torch.load(data_path)  # list of (positions, adjacency)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        positions, adjacency = self.data[idx]
        # positions: shape [time_steps, node_amt, 3]
        # adjacency: shape [time_steps, node_amt, node_amt]
        return positions, adjacency


def adjacency_to_edge_index(adj_t: torch.Tensor):
    """
    Converts a single adjacency matrix (node_amt x node_amt) into
    an edge_index of shape [2, num_edges].
    Here, adj_t is assumed to be 0/1 and symmetric.
    
    We only take upper-triangular or just the non-zero entries
    to avoid duplicating edges. GCNConv can handle duplicates,
    but it's neater to remove them.
    """
    # (node_i, node_j) for all 1-entries
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index


def collate_fn(batch):
    """
    Collates a list of (positions, adjacency) into a single batch.
    batch: list of tuples where each tuple is (positions, adjacency).

    We'll produce:
        - x_batch of shape [batch_size, num_nodes, num_timesteps, input_dim]
        - edge_indices_list: a list (of length num_timesteps) of edge_index
          for the entire batch. BUT typically, you can't just merge adjacency
          across different examples if you want them in one big graph. 

    For demonstration, we show a simpler approach: process each sample
    in the batch *separately*, which is a bit unusual for a GCN, but 
    simpler to illustrate. We'll just return them as a list-of-samples 
    and handle them one by one in the training loop. 
    """
    return batch  # We'll handle them in the training step manually.


class InfoNCELoss(nn.Module):
    """
    A simple InfoNCE-like loss for node embeddings:
      - anchor: node i
      - positive: a node j from the same group
      - negatives: all nodes k from different groups

    For each anchor i, we pick exactly one positive j from the same group (randomly)
    and treat the rest as negatives. 
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, groups):
        """
        Args:
            embeddings: [batch_size, num_nodes, embed_dim]
            groups: [batch_size, num_nodes]  (group indices for each node)

        Returns:
            scalar loss (tensor)
        """
        # We'll do this for each sample in the batch separately and then average.
        batch_size, num_nodes, embed_dim = embeddings.shape
        device = embeddings.device
        
        total_loss = 0.0
        for b in range(batch_size):
            emb_b = embeddings[b]   # shape: [num_nodes, embed_dim]
            grp_b = groups[b]       # shape: [num_nodes]
            
            # For each node i, we pick one positive j from the same group (if possible).
            loss_b = 0.0
            count_b = 0
            
            for i in range(num_nodes):
                group_i = grp_b[i].item()
                same_group_idxs = (grp_b == group_i).nonzero(as_tuple=True)[0]
                # Exclude i itself
                same_group_idxs = same_group_idxs[same_group_idxs != i]
                
                if len(same_group_idxs) < 1:
                    # No positive available; skip
                    continue
                
                # Pick exactly one positive at random:
                j = same_group_idxs[torch.randint(len(same_group_idxs), (1,))]
                
                # Anchor and positive embeddings
                anchor = emb_b[i].unsqueeze(0)       # [1, embed_dim]
                positive = emb_b[j].unsqueeze(0)     # [1, embed_dim]
                
                # Compute similarity with all nodes to anchor
                sim_all = torch.mm(anchor, emb_b.t()) / self.temperature
                # shape: [1, num_nodes] (dot product with every node)
                
                # We'll classify the "positive index" among these num_nodes as the correct class
                # The cross-entropy label is the index of j.
                label = j  # j is already a single-element tensor (the index)
                
                # Reshape sim_all and label to fit CrossEntropyLoss expectation:
                # CE expects shape [batch_size, n_classes], label [batch_size]
                # Here we have "batch_size=1" and "n_classes=num_nodes".
                loss_i = self.ce(sim_all, label)
                loss_b += loss_i
                count_b += 1
            
            if count_b > 0:
                loss_b /= count_b
            total_loss += loss_b
        
        return total_loss / batch_size


def train_one_epoch(model, dataloader, optimizer, device, infonce_loss_fn):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, sample_list in enumerate(dataloader):
        # sample_list is a list of (positions, adjacency), 
        # one for each item in the batch. We'll accumulate the loss.
        # Then backprop once per batch.

        # If you want to handle each sample in the batch in a single big graph,
        # you'd have to combine them carefully. For simplicity, we do a loop:
        
        batch_embeddings = []
        batch_groups = []

        for (positions, adjacency) in sample_list:
            # positions: [time_steps, node_amt, 3]
            # adjacency: [time_steps, node_amt, node_amt]
            
            time_steps, node_amt, _ = positions.shape
            
            # The group ID is in positions[..., 2], which is the same for all timesteps.
            # We'll just use positions[0, :, 2] as the group assignments for each node.
            group_ids = positions[0, :, 2].long()
            
            # Build the feature tensor x of shape [1, node_amt, time_steps, input_dim=2 or 3?]
            # input_dim could be 3 if we keep (x, y, group_id). 
            # If we only want (x, y), set input_dim=2 and remove group_id from x.
            # Let's assume we want all 3 for now:
            #   x -> [batch_size=1, node_amt, time_steps, 3]
            
            # Rearrange positions from [time_steps, node_amt, 3] to [node_amt, time_steps, 3]
            positions_transposed = positions.permute(1, 0, 2)  # shape: [node_amt, time_steps, 3]
            x = positions_transposed.unsqueeze(0).to(device)   # [1, node_amt, time_steps, 3]
            
            # Convert adjacency to list of edge_indices
            edge_indices = []
            for t in range(time_steps):
                adj_t = adjacency[t]
                # Convert to edge_index
                edge_index_t = adjacency_to_edge_index(adj_t)
                edge_index_t = edge_index_t.to(device)
                edge_indices.append(edge_index_t)
            
            # Forward pass
            emb = model(x, edge_indices)  # shape: [1, node_amt, output_dim]
            # We'll store this for the InfoNCE
            batch_embeddings.append(emb.squeeze(0))  # shape: [node_amt, output_dim]
            batch_groups.append(group_ids.to(device))  # shape: [node_amt]
        
        # Now we have a "batch" of embeddings: len(sample_list) items
        # We can stack them into [batch_size, num_nodes, output_dim]
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # [B, node_amt, output_dim]
        batch_groups = torch.stack(batch_groups, dim=0)  # [B, node_amt]
        
        # Compute InfoNCE loss
        loss = infonce_loss_fn(batch_embeddings, batch_groups)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, infonce_loss_fn):
    model.eval()
    epoch_loss = 0.0
    
    for batch_idx, sample_list in enumerate(dataloader):
        batch_embeddings = []
        batch_groups = []

        for (positions, adjacency) in sample_list:
            time_steps, node_amt, _ = positions.shape
            group_ids = positions[0, :, 2].long()
            positions_transposed = positions.permute(1, 0, 2).unsqueeze(0).to(device)
            
            # Convert adjacency to list of edge_indices
            edge_indices = []
            for t in range(time_steps):
                adj_t = adjacency[t]
                edge_index_t = adjacency_to_edge_index(adj_t).to(device)
                edge_indices.append(edge_index_t)
            
            emb = model(positions_transposed, edge_indices)
            batch_embeddings.append(emb.squeeze(0))
            batch_groups.append(group_ids.to(device))
        
        batch_embeddings = torch.stack(batch_embeddings, dim=0)
        batch_groups = torch.stack(batch_groups, dim=0)
        loss = infonce_loss_fn(batch_embeddings, batch_groups)
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='test_data.pt',
                        help="Path to your training dataset .pt file.")
    parser.add_argument('--val_path', type=str, default='val_data.pt',
                        help="Path to your validation dataset .pt file.")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--temp', type=float, default=0.1, 
                        help="Temperature for InfoNCE.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load Datasets ---
    train_dataset = GCNDataset(args.train_path)
    val_dataset = GCNDataset(args.val_path)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)
    
    # We know from genEpisodes.py: 
    # node_amt=400, group_amt=4, time_steps=20, input_dim=3
    # But you might want to be more flexible by *inspecting* one sample from the dataset.
    sample_positions, _ = train_dataset[0]
    time_steps, node_amt, feat_dim = sample_positions.shape  # e.g. 20, 400, 3
    
    # Create model
    # input_dim=3 (x, y, group), output_dim let's pick something, e.g. 32
    # hidden_dim is adjustable
    model = TemporalGCN(
        input_dim=feat_dim,
        output_dim=32,  # this is your embedding dim
        num_nodes=node_amt,
        num_timesteps=time_steps,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # InfoNCE Loss
    infonce_loss_fn = InfoNCELoss(temperature=args.temp)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, infonce_loss_fn)
        val_loss = validate_one_epoch(model, val_loader, device, infonce_loss_fn)
        
        print(f"Epoch [{epoch}/{args.epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        
        # You can implement a simple early-stopping or model checkpointing:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("  [*] Model saved.")
    
    print("Training completed. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()

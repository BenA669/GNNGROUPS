import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from model import TemporalGCN 


class GCNDataset(Dataset):
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
    # (node_i, node_j) for all 1-entries
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index


def collate_fn(batch):
    return batch


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, groups):
        device = embeddings.device
        B, N, emb_dim = embeddings.shape

        # Flatten (B, N) -> (B*N)
        flat_emb = embeddings.view(B * N, emb_dim)        # [B*N, emb_dim]
        flat_groups = groups.view(B * N)                  # [B*N]

        # Build dict: group_id -> list of all indices with that group
        from collections import defaultdict
        group_dict = defaultdict(list)
        for idx in range(B * N):
            g = flat_groups[idx].item()
            group_dict[g].append(idx)

        anchor_idx_list = []
        pos_idx_list = []
        neg_idx_list = []

        # We'll create a global range of indices for quick complement sampling
        all_indices = torch.arange(B * N, device=device)

        for g, idxs in group_dict.items():
            idxs = torch.tensor(idxs, device=device)

            # We can't form a positive pair if there's < 2 members in this group
            if idxs.size(0) < 2:
                continue

            # Random permutation of the indices in this group
            perm = idxs[torch.randperm(idxs.size(0))]

            # "Shift" the permutation by 1 to get a positive partner for each anchor
            # e.g. perm = [7, 10, 5], pos = [10, 5, 7]
            pos = torch.roll(perm, shifts=1, dims=0)

            anchor_idx_list.append(perm)
            pos_idx_list.append(pos)

            # Now sample negatives from the complement of this group
            complement_mask = (flat_groups != g)
            complement_indices = all_indices[complement_mask]
            if complement_indices.numel() == 0:
                # If there's no complement (i.e., *all* nodes are in group g),
                # we can't sample negatives for this group. Skip them.
                continue

            # For each anchor in this group, choose 1 negative from the complement
            neg = complement_indices[
                torch.randint(
                    high=complement_indices.size(0),
                    size=(idxs.size(0),),
                    device=device
                )
            ]
            neg_idx_list.append(neg)

        if len(anchor_idx_list) == 0:
            # Edge case: no valid anchors => return zero or something safe
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Concatenate all anchors, positives, negatives
        anchor_idx = torch.cat(anchor_idx_list, dim=0)  # [K]
        pos_idx    = torch.cat(pos_idx_list,    dim=0)  # [K]
        neg_idx    = torch.cat(neg_idx_list,    dim=0)  # [K]
        
        # Gather embeddings
        anchor_emb = flat_emb[anchor_idx]   # [K, emb_dim]
        pos_emb    = flat_emb[pos_idx]      # [K, emb_dim]
        neg_emb    = flat_emb[neg_idx]      # [K, emb_dim]

        # Compute dot products for positives and negatives
        pos_score = torch.sum(anchor_emb * pos_emb, dim=1) / self.temperature  # [K]
        neg_score = torch.sum(anchor_emb * neg_emb, dim=1) / self.temperature  # [K]

        # Pack scores into logits: [pos_score, neg_score]
        logits = torch.stack([pos_score, neg_score], dim=1)  # [K, 2]

        # Labels are all zero (i.e., pos_score is the "correct" logit)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

        # Compute 2-class cross-entropy
        loss = self.ce(logits, labels)
        return loss


def train_one_epoch(model, dataloader, optimizer, device, infonce_loss_fn):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, sample_list in enumerate(dataloader):
        # sample_list is a list of (positions, adjacency), 
        # one for each item in the batch. We'll accumulate the loss.
        # Then backprop once per batch.
        
        batch_embeddings = []
        batch_groups = []

        for (positions, adjacency) in sample_list:
            # positions: [time_steps, node_amt, 3]
            # adjacency: [time_steps, node_amt, node_amt]
            
            time_steps, node_amt, _ = positions.shape
            
            
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
    
    for epoch in tqdm(range(1, args.epochs+1)):
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

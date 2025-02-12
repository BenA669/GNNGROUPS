import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from model import TemporalGCN 
from makeEpisode import getEgo
from torch.nn.utils.rnn import pad_sequence
from GraphDataset import GCNDataset, collate_fn

# class GCNDataset(Dataset):
#     def __init__(self, data_path):
#         super().__init__()
#         self.data = torch.load(data_path)
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         (positions, adjacency, edge_indices, 
#         ego_idx, ego_positions, ego_adjacency, 
#         ego_edge_indices, EgoMask) = self.data[idx]
#         # Convert everything to tensors
#         return (positions, adjacency, edge_indices, 
#                 ego_idx, ego_positions, ego_adjacency, 
#                 ego_edge_indices, EgoMask)

def adjacency_to_edge_index(adj_t: torch.Tensor):
    # (node_i, node_j) for all 1-entries
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index


# def collate_fn(batch):
#     # Unzip the batch (each sample is a tuple)
#     positions, adjacency, edge_indices, ego_idx, ego_positions, ego_adjacency, ego_edge_indices, ego_mask = zip(*batch)

#     print(type(ego_positions))
#     exit()
    
#     # Stack the ego_positions (and any other elements you want to batch)
#     positions_batch = torch.stack(positions, dim=0) # [batch_size, time_stamp, node_amt, 3]
#     ego_mask_batch = torch.stack(ego_mask, dim=0)

#     big_batch_edges = []
#     big_batch_positions = []
#     # edge_indicies = [batch, timestamp, [2, N]]
#     B = len(edge_indices)
#     T = len(edge_indices[0])
#     max_nodes = positions_batch.size(dim=2)
#     # print("Batch size: {}".format(B))
#     for t in range(T):
#         edges_at_t = []
#         positions_at_t = []
#         for b in range(B):
#             # Get the edge-index for batch element b at timestamp t.
#             e = edge_indices[b][t]  # shape [2, N_b]
#             p = positions_batch[b, t, :, :2] # shape [node_amt, 3]

#             # Offset the node indices so that nodes in batch b get indices in [b*max_nodes, (b+1)*max_nodes)
#             e_offset = e + b * max_nodes
#             p_offset = p + b*max_nodes
            
#             positions_at_t.append(p_offset)
#             edges_at_t.append(e_offset)
#         # Concatenate all batches’ edge indices for this timestamp along the second dimension.
#         combined_edges = torch.cat(edges_at_t, dim=1).to(torch.device('cuda:0'))  # shape [2, total_edges_at_t]
#         big_batch_edges.append(combined_edges)

#         combined_pos = torch.cat(positions_at_t, dim=0).to(torch.device('cuda:0'))
#         big_batch_positions.append(combined_pos)

#     # print(type(big_batch_edges[0]))
#     # print(len(big_batch_edges[1][0]))
    
#     # You can also stack or combine the other items if needed.
#     # For now, we'll return all components in a dictionary:
#     return {
#         'positions': positions_batch,
#         'adjacency': adjacency,
#         'edge_indices': edge_indices,
#         'ego_idx': ego_idx,
#         'ego_positions': ego_positions,  # This now has shape [batch, timesteps, node_amt, 3]
#         'ego_adjacency': ego_adjacency,
#         'ego_edge_indices': ego_edge_indices,
#         'ego_mask_batch': ego_mask_batch,
#         'big_batch_edges': big_batch_edges,
#         'big_batch_positions': big_batch_positions,
#     }

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, groups, mask=None):
        """
        Accepts either:
          - embeddings: a tensor of shape [B, N, emb_dim] and groups: a tensor of shape [B, N], or
          - embeddings: a list of tensors, each of shape [num_nodes, emb_dim],
            and groups: a list of tensors, each of shape [num_nodes].
        The function concatenates the list case into flat tensors, so that negatives
        can be sampled across all nodes in the batch.
        """
        # If embeddings are provided as a list, concatenate them along the node dimension.
        if isinstance(embeddings, list):
            # Each element in the list is assumed to be of shape [num_nodes, emb_dim]
            flat_emb = torch.cat(embeddings, dim=0)  # [total_nodes, emb_dim]
            flat_groups = torch.cat(groups, dim=0)     # [total_nodes]
        else:
            # Assume embeddings is a tensor of shape [B, N, emb_dim]
            B, N, emb_dim = embeddings.shape
            flat_emb = embeddings.view(B * N, emb_dim)
            flat_groups = groups.reshape(B * N)
        
        device = flat_emb.device
        total_nodes = flat_emb.shape[0]

        # Build a dictionary mapping group id to the list of indices with that group.
        from collections import defaultdict
        group_dict = defaultdict(list)
        for idx in range(total_nodes):
            g = flat_groups[idx].item()
            group_dict[g].append(idx)

        anchor_idx_list = []
        pos_idx_list = []
        neg_idx_list = []
        # Create a global tensor of indices for negative sampling.
        all_indices = torch.arange(total_nodes, device=device)

        for g, idxs in group_dict.items():
            idxs_tensor = torch.tensor(idxs, device=device)
            # Skip groups with fewer than 2 members.
            if idxs_tensor.size(0) < 2:
                continue

            # Shuffle the indices and pair each index with the next one (cyclically).
            perm = idxs_tensor[torch.randperm(idxs_tensor.size(0))]
            pos = torch.roll(perm, shifts=1, dims=0)

            anchor_idx_list.append(perm)
            pos_idx_list.append(pos)

            # Sample one negative for each anchor from outside the current group.
            complement_mask = (flat_groups != g)
            complement_indices = all_indices[complement_mask]
            if complement_indices.numel() == 0:
                continue

            neg = complement_indices[
                torch.randint(
                    high=complement_indices.size(0),
                    size=(idxs_tensor.size(0),),
                    device=device
                )
            ]
            neg_idx_list.append(neg)

        # Handle the edge-case where no valid positive pairs are found.
        if len(anchor_idx_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Concatenate all anchor, positive, and negative indices.
        anchor_idx = torch.cat(anchor_idx_list, dim=0)
        pos_idx    = torch.cat(pos_idx_list,    dim=0)
        neg_idx    = torch.cat(neg_idx_list,    dim=0)
        
        # Gather the embeddings.
        anchor_emb = flat_emb[anchor_idx]   # [K, emb_dim]
        pos_emb    = flat_emb[pos_idx]      # [K, emb_dim]
        neg_emb    = flat_emb[neg_idx]      # [K, emb_dim]

        # Compute dot products for positives and negatives, scaled by temperature.
        pos_score = torch.sum(anchor_emb * pos_emb, dim=1) / self.temperature  # [K]
        neg_score = torch.sum(anchor_emb * neg_emb, dim=1) / self.temperature  # [K]

        # Create logits and labels.
        logits = torch.stack([pos_score, neg_score], dim=1)  # [K, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

        # Compute and return the cross-entropy loss.
        loss = self.ce(logits, labels)
        return loss

def train_one_epoch_better(model, dataloader, optimizer, device, infonce_loss_fn):
    model.train()
    epoch_loss=0.0
    print("Training epoch")
    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions']
        groups = positions[:, 0, :, 2]
        ego_mask_batch = batch['ego_mask_batch']
        big_batch_edges = batch['big_batch_edges']
        big_batch_positions = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batch_adjacency']


        emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)
        
        loss = infonce_loss_fn(emb, groups)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



@torch.no_grad()
def validate_one_epoch(model, dataloader, device, infonce_loss_fn):
    model.eval()
    epoch_loss = 0.0
    
    print("Validating")
    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions']
        groups = positions[:, 0, :, 2]
        ego_mask_batch = batch['ego_mask_batch']
        big_batch_edges = batch['big_batch_edges']
        big_batch_positions = batch['big_batch_positions']
        big_batch_adjacency = batch['big_batch_adjacency']


        emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)
        
        loss = infonce_loss_fn(emb, groups)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='test_data_Ego.pt',
                        help="Path to your training dataset .pt file.")
    parser.add_argument('--val_path', type=str, default='val_data_Ego.pt',
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
    node_amt=400
    group_amt=4
    time_steps=20
    input_dim=2
    # But you might want to be more flexible by *inspecting* one sample from the dataset.
    # sample_positions, _ = train_dataset[0][0]['positions']

    # print("smpale shale: {}".format(sample_positions.shape))
    # time_steps, node_amt, feat_dim = sample_positions.shape  # e.g. 20, 400, 3
    
    # Create model
    # input_dim=3 (x, y, group), output_dim let's pick something, e.g. 32
    # hidden_dim is adjustable
    model = TemporalGCN(
        input_dim=input_dim,
        output_dim=16,  # this is your embedding dim
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
        # train_loss = train_one_epoch(model, train_loader, optimizer, device, infonce_loss_fn)
        train_loss = train_one_epoch_better(model, train_loader, optimizer, device, infonce_loss_fn)
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

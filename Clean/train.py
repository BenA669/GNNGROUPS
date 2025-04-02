import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from model import TemporalGCN 
from makeEpisode import getEgo
from torch.nn.utils.rnn import pad_sequence
from datasetEpisode import GCNDataset, collate_fn


def adjacency_to_edge_index(adj_t: torch.Tensor):
    # (node_i, node_j) for all 1-entries
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELossOld(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature (float): Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, groups, mask):
        """
        Args:
            embeddings (Tensor): Tensor of shape [B, N, emb_dim].
            mask (Tensor): Binary mask of shape [B, N] where 1 indicates a valid token.
            groups (Tensor): Tensor of shape [B, N] with group labels.
        
        Returns:
            loss (Tensor): Scalar tensor representing the InfoNCE loss.
        """
        B, N, emb_dim = embeddings.shape
        groups = groups.to(embeddings.device)
        mask = mask.to(embeddings.device)
        
        # print(f"embed: {embeddings.get_device()}")
        # print(f"groups: {groups.get_device()}")
        # print(f"mask: {mask.get_device()}")
        # Normalize embeddings to unit length for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # [B, N, emb_dim]
        
        # Compute similarity matrix [B, N, N] via dot product
        sim_matrix = torch.matmul(embeddings, embeddings.transpose(1, 2))  # [B, N, N]
        sim_matrix = sim_matrix / self.temperature  # scale similarities
        
        # Create a mask for valid tokens (expand for pairwise comparisons)
        valid_mask = mask.bool()  # [B, N]
        valid_mask_i = valid_mask.unsqueeze(2)  # [B, N, 1]
        valid_mask_j = valid_mask.unsqueeze(1)  # [B, 1, N]
        valid_pair_mask = valid_mask_i & valid_mask_j  # [B, N, N] valid pairs
        
        # Identify positive pairs: same group and not comparing the token with itself.
        group_equal = (groups.unsqueeze(2) == groups.unsqueeze(1))  # [B, N, N]
        diag_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device).unsqueeze(0)  # [B, N, N]
        positive_mask = group_equal & ~diag_mask  # exclude self comparisons
        
        # Only consider valid tokens in the positives as well
        positive_mask = positive_mask & valid_pair_mask
        
        # Negatives: valid pairs that are not in the positive set
        negative_mask = valid_pair_mask & (~positive_mask)
        
        # For each anchor, we want:
        #   logit_denominator = logsumexp( sim(anchor, all valid tokens) )
        #   logit_numerator   = logsumexp( sim(anchor, positive tokens) )
        # We first mask out invalid comparisons by setting them to a very low value.
        logits = sim_matrix.masked_fill(~valid_pair_mask, -1e9)
        
        # Compute denominator: over all valid comparisons per anchor.
        denom = torch.logsumexp(logits, dim=-1)  # [B, N]
        
        # For the numerator, mask out all non-positive pairs.
        pos_logits = logits.masked_fill(~positive_mask, -1e9)
        num = torch.logsumexp(pos_logits, dim=-1)  # [B, N]
        
        # InfoNCE loss per valid token: -log(probability of positives)
        loss_per_token = -(num - denom)  # [B, N]
        
        # Only average over tokens that have at least one positive candidate.
        has_positive = positive_mask.any(dim=-1)  # [B, N]
        loss = loss_per_token[has_positive]
        if loss.numel() == 0:
            # In case no valid positive pairs exist, return zero loss.
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return loss.mean()
    

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        """
        InfoNCE Loss for contrastive learning based on group labels.
        
        Args:
            temperature (float): Temperature scaling factor.
            reduction (str): Reduction mode. Currently only 'mean' is supported.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, embeddings, groups, mask=None):
        """
        Compute the InfoNCE loss.

        Args:
            embeddings (torch.Tensor): Tensor of shape [B, N, D] where B is batch size,
                N is number of nodes (or nodes in the ego network) and D is embedding dimension.
            groups (torch.Tensor): Tensor of shape [B, N] containing group labels for each node.
            mask (torch.Tensor, optional): Boolean tensor of shape [B, N] indicating which nodes
                should be included in the loss computation. If None, all nodes are used.

        Returns:
            torch.Tensor: The computed InfoNCE loss (a scalar).
        """
        batch_loss = 0.0
        total_anchors = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
        B = embeddings.size(0)
        # Process each sample in the batch separately.
        for b in range(B):
            # Select embeddings and corresponding groups for this sample.
            z = embeddings[b]   # shape: [N, D]
            g = groups[b]       # shape: [N]
            if mask is not None:
                mask_b = mask[b]
                z = z[mask_b]
                g = g[mask_b]
            
            N = z.size(0)
            if N < 2:
                continue  # Skip if fewer than 2 nodes are available.
            
            # Normalize the embeddings so that cosine similarity equals dot product.
            z = F.normalize(z, p=2, dim=1)
            # Compute similarity matrix and scale by temperature.
            sim = torch.matmul(z, z.t()) / self.temperature
            # Remove self-similarity by setting diagonal to -inf (so exp(-inf)=0)
            diag_mask = torch.eye(N, device=z.device).bool()
            sim.masked_fill_(diag_mask, -float('inf'))
            
            # Create a binary mask for positives: nodes with the same group label (excluding self).
            # Expand dims to compare every pair.
            g = g.view(-1)
            positive_mask = (g.unsqueeze(0) == g.unsqueeze(1))  # shape: [N, N]
            positive_mask = positive_mask.fill_diagonal_(False)
            
            # Compute exponentials of the similarities.
            exp_sim = torch.exp(sim)
            # For each anchor, denominator sums over all other nodes.
            denominator = exp_sim.sum(dim=1)  # shape: [N]
            # Numerator: sum over positive nodes.
            positive_mask = positive_mask.to(device)
            # print(exp_sim.device)
            # print(positive_mask.device)
            
            numerator = (exp_sim * positive_mask.float()).sum(dim=1)  # shape: [N]
            
            # Identify anchors that have at least one positive.
            valid = numerator > 0
            if valid.sum() == 0:
                continue

            # Compute the loss for valid anchors.
            loss_b = -torch.log(numerator[valid] / denominator[valid])
            batch_loss += loss_b.sum()
            total_anchors += valid.sum()
        
        if total_anchors == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        loss = batch_loss / total_anchors
        return loss


def train_one_epoch_better(model, dataloader, optimizer, device, infonce_loss_fn):
    model.train()
    epoch_loss=0.0
    print("Training epoch")
    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions']
        groups = positions[:, 0, :, 2]
        ego_mask_batch = batch['ego_mask_batch'] # Shape: (Batch, Timestep, Node Amt)
        ego_mask = ego_mask_batch.any(dim=1)  # shape: [B, N]


        # emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)
        emb = model(batch)
        
        # loss = infonce_loss_fn(emb, groups)
        loss = infonce_loss_fn(emb, groups, mask=ego_mask)
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
        ego_mask = ego_mask_batch.any(dim=1)  # shape: [B, N]


        # emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)
        emb = model(batch)
        
        loss = infonce_loss_fn(emb, groups, mask=ego_mask)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='test_data_Ego_2hop.pt',
                        help="Path to your training dataset .pt file.")
    parser.add_argument('--val_path', type=str, default='val_data_Ego_2hop.pt',
                        help="Path to your validation dataset .pt file.")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
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
    node_amt=200
    group_amt=3
    time_steps=10
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
    infonce_loss_fn = InfoNCELossOld(temperature=args.temp)
    
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

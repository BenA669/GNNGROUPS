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

class InfoNCELoss(nn.Module):
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
        
        loss = infonce_loss_fn(emb, groups, ego_mask)
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

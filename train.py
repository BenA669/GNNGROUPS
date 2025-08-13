import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from datasetEpisode import GCNDataset, collate_fn
import torch.nn.functional as F
import configparser
from tqdm import tqdm
from configReader import read_config

def adjacency_to_edge_index(adj_t: torch.Tensor):
    edge_index = adj_t.nonzero().t().contiguous()  # shape [2, E]
    return edge_index

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, embeddings, groups, mask=None):
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
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        loss = batch_loss / total_anchors
        return loss


def train_one_epoch_better(model, dataloader, optimizer, device, infonce_loss_fn):
    model.train()
    epoch_loss=0.0
    print("Training epoch")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        positions = batch['positions']
        groups = positions[:, 0, :, 2]
        ego_mask_batch = batch['ego_mask_batch'] # Shape: (Batch, Timestep, Node Amt)
        ego_mask = ego_mask_batch.any(dim=1)  # shape: [B, N]

        # Get embeddings with shape [B, max_nodes, T, hidden_dim]
        emb = model(batch)

        # Compute loss at each timestep and average
        B, max_nodes, T, D = emb.shape
        loss = 0.0
        for t in range(T):
            loss_t = infonce_loss_fn(emb[:, :, t, :], groups, mask=ego_mask)
            loss += loss_t
        loss = loss / T

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
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        positions = batch['positions']
        groups = positions[:, 0, :, 2]
        ego_mask_batch = batch['ego_mask_batch']
        ego_mask = ego_mask_batch.any(dim=1)  # shape: [B, N]

        emb = model(batch)  # shape: [B, max_nodes, T, hidden_dim]

        # Compute loss over all timesteps and average.
        B, max_nodes, T, D = emb.shape
        loss = 0.0
        for t in range(T):
            loss_t = infonce_loss_fn(emb[:, :, t, :], groups, mask=ego_mask)
            loss += loss_t
        loss = loss / T

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



if __name__ == "__main__":
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    print(dataset_cfg["dir_path"])     # → "./attention_4/"
    print(dataset_cfg["dataset_name"]) # → "attn_4_BROKERNG"

    print(dataset_cfg["train_path"])   # → "./attention_4/attn_4_BROKERNG_train.pt"
    print(dataset_cfg["val_path"])     # → "./attention_4/attn_4_BROKERNG_val.pt"

    dir_path = dataset_cfg["dir_path"]

    val_dataset = GCNDataset(dataset_cfg["val_path"])
    train_dataset = GCNDataset(dataset_cfg["train_path"])

    batch_size = training_cfg["batch_size"]
    temp = training_cfg["temp"]
    lr = training_cfg["learning_rate"]
    epochs = training_cfg["epochs"]
    model_name = training_cfg["model_name_pt"]
    model_save = "{}{}".format(dir_path, model_name)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)

    model = getModel(eval=False)
    
    # InfoNCE Loss
    infonce_loss_fn = InfoNCELoss(temperature=temp)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch_better(model, train_loader, optimizer, device, infonce_loss_fn)
        val_loss = validate_one_epoch(model, val_loader, device, infonce_loss_fn)
        
        print(f"Epoch [{epoch}/{epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print("  [*] Model saved.")
    
    print("Training completed. Best val loss:", best_val_loss)

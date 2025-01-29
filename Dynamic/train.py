# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import TemporalGCN
from makeDataset import makeDatasetDynamic
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import math

class DynamicGraphDataset(Dataset):
    def __init__(self, num_episodes, time_steps=20, node_amt=400, group_amt=4, std_dev=1,
                 speed_min=0.01, speed_max=0.5, threshold=0.1):
        self.num_episodes = num_episodes
        self.time_steps = time_steps
        self.node_amt = node_amt
        self.group_amt = group_amt
        self.std_dev = std_dev
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.threshold = threshold
        
        self.graphs = [makeDatasetDynamic(
            node_amt=self.node_amt,
            group_amt=self.group_amt,
            std_dev=self.std_dev,
            speed_min=self.speed_min,
            speed_max=self.speed_max,
            time_steps=self.time_steps,
            intra_prob=0.05,
            inter_prob=0.001
        ) for _ in range(self.num_episodes)]

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        all_positions, _, all_positions_cpu, adj_matrix_cpu = self.graphs[idx]
        labels = all_positions_cpu[-1, :, 2].long()  # Assuming labels are based on groups

        # Extract node coordinates for each timestep
        x_seq = all_positions_cpu[:, :, 0].unsqueeze(-1)  # [time_steps, node_amt, 1]
        y_seq = all_positions_cpu[:, :, 1].unsqueeze(-1)  # [time_steps, node_amt, 1]
        coords_seq = torch.cat([x_seq, y_seq], dim=2)  # [time_steps, node_amt, 2]

        # Dynamic adjacency based on distance threshold for each timestep
        adj_seq = []
        for t in range(self.time_steps):
            coords = coords_seq[t]  # [node_amt, 2]
            # Compute pairwise distances
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [node_amt, node_amt, 2]
            dist = torch.norm(diff, dim=2)  # [node_amt, node_amt]
            adj = (dist < self.threshold).float()
            adj.fill_diagonal_(0)  # Remove self-loops
            adj_seq.append(adj)
        
        adj_seq = torch.stack(adj_seq, dim=0)  # [time_steps, node_amt, node_amt]
        
        return coords_seq, adj_seq, labels

def findRightPerm(predicted_labels, labels):
    from scipy.optimize import linear_sum_assignment

    # Convert to numpy
    predicted = predicted_labels.cpu().numpy()
    true = labels.cpu().numpy()

    # Compute the confusion matrix
    D = max(predicted.max(), true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(predicted.size):
        w[predicted[i], true[i]] +=1

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in predicted])

    accuracy = np.mean(mapped_pred == true)
    return mapped_pred, accuracy

def InfoNCELoss(embeddings, labels, temperature=0.1):
    """
    embeddings: [batch_size, num_nodes, embedding_dim]
    labels: [batch_size, num_nodes]
    """
    device = embeddings.device
    batch_size, num_nodes, embed_dim = embeddings.size()

    embeddings = embeddings.view(batch_size * num_nodes, embed_dim)
    labels = labels.view(batch_size * num_nodes)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [BN, BN]

    # Create labels mask
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # Remove self-similarity
    mask.fill_diagonal_(0)

    # Compute positives and negatives
    positives = similarity_matrix * mask
    negatives = similarity_matrix * (1 - mask)

    # For numerical stability
    logits = similarity_matrix / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Labels for contrastive loss
    labels_contrastive = torch.arange(batch_size * num_nodes).to(device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels_contrastive)

    return loss

def train():
    # Hyperparameters
    num_episodes = 1000
    time_steps = 20
    node_amt = 400
    group_amt = 4
    std_dev = 1
    speed_min = 0.01
    speed_max = 0.5
    threshold = 0.1
    batch_size = 10
    epochs = 100
    learning_rate = 0.001
    weight_decay = 5e-4

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = DynamicGraphDataset(
        num_episodes=num_episodes,
        time_steps=time_steps,
        node_amt=node_amt,
        group_amt=group_amt,
        std_dev=std_dev,
        speed_min=speed_min,
        speed_max=speed_max,
        threshold=threshold
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, optimizer, and loss
    model = TemporalGCN(input_dim=2, hidden_dim=64, embedding_dim=16, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (coords_seq, adj_seq, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")):
            coords_seq = coords_seq.to(device)  # [batch_size, time_steps, num_nodes, 2]
            adj_seq = adj_seq.to(device)        # [batch_size, time_steps, num_nodes, num_nodes]
            labels = labels.to(device)          # [batch_size, num_nodes]

            optimizer.zero_grad()
            embeddings = model(coords_seq, adj_seq)  # [batch_size, num_nodes, 16]
            loss = InfoNCELoss(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        # Optionally, add evaluation and checkpointing here

    # Save the model
    torch.save(model.state_dict(), 'temporal_gcn_model.pth')
    print("Model saved as 'temporal_gcn_model.pth'.")

if __name__ == '__main__':
    train()

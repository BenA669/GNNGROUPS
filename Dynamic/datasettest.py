import torch
from torch.utils.data import Dataset, DataLoader

class GCNDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        (positions, adjacency, edge_indices, 
        ego_idx, ego_positions, ego_adjacency, 
        ego_edge_indices, EgoMask) = self.data[idx]
        # Convert everything to tensors
        return (positions, adjacency, edge_indices, 
                ego_idx, ego_positions, ego_adjacency, 
                ego_edge_indices, EgoMask)

def collate_fn(batch):
    # Unzip the batch (each sample is a tuple)
    positions, adjacency, edge_indices, ego_idx, ego_positions, ego_adjacency, ego_edge_indices, ego_mask = zip(*batch)
    
    # Stack the ego_positions (and any other elements you want to batch)
    positions_batch = torch.stack(positions, dim=0) # [batch_size, time_stamp, node_amt, 3]
    ego_mask_batch = torch.stack(ego_mask, dim=0)

    big_batch_edges = []
    big_batch_positions = []
    # edge_indicies = [batch, timestamp, [2, N]]
    B = len(edge_indices)
    T = len(edge_indices[0])
    max_nodes = positions_batch.size(dim=2)
    for t in range(T):
        edges_at_t = []
        positions_at_t = []
        for b in range(B):
            # Get the edge-index for batch element b at timestamp t.
            e = edge_indices[b][t]  # shape [2, N_b]
            p = positions_batch[b, t, :, :2] # shape [node_amt, 3]

            # Offset the node indices so that nodes in batch b get indices in [b*max_nodes, (b+1)*max_nodes)
            e_offset = e + b * max_nodes
            p_offset = p + b*max_nodes
            
            positions_at_t.append(p_offset)
            edges_at_t.append(e_offset)
        # Concatenate all batches’ edge indices for this timestamp along the second dimension.
        combined_edges = torch.cat(edges_at_t, dim=1).to(torch.device('cuda:0'))  # shape [2, total_edges_at_t]
        big_batch_edges.append(combined_edges)

        combined_pos = torch.cat(positions_at_t, dim=0).to(torch.device('cuda:0'))
        big_batch_positions.append(combined_pos)

    # print(type(big_batch_edges[0]))
    # print(len(big_batch_edges[1][0]))
    
    # You can also stack or combine the other items if needed.
    # For now, we'll return all components in a dictionary:
    return {
        'positions': positions_batch,
        'adjacency': adjacency,
        'edge_indices': edge_indices,
        'ego_idx': ego_idx,
        'ego_positions': ego_positions,  # This now has shape [batch, timesteps, node_amt, 3]
        'ego_adjacency': ego_adjacency,
        'ego_edge_indices': ego_edge_indices,
        'ego_mask_batch': ego_mask_batch,
        'big_batch_edges': big_batch_edges,
        'big_batch_positions': big_batch_positions,
    }

if __name__ == '__main__':
    # Create dataset
    dataset = GCNDataset('test_data_Ego.pt')

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        print(batch['positions'].shape)
        break

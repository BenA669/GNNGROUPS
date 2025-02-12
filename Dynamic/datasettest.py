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
    B = len(edge_indices)
    T = len(edge_indices[0])    

    # print(ego_positions[0].shape)

    # ego_positions [batch, timestamp, node_amt, (x,y,Group)]
    
    # Stack the ego_positions (and any other elements you want to batch)
    positions_batch = torch.stack(positions, dim=0) # [batch_size, time_stamp, node_amt, 3]
    adjacency_batch = torch.stack(adjacency, dim=0) # [batchsize, time_stamp, node_amt, node_amt]
    ego_mask_batch = torch.stack(ego_mask, dim=0)

    max_nodes = positions_batch.size(dim=2)
    
    
    

    # [batchsize, time_stamp, node_amt, node_amt] -> [time_stamp, node_amt*batchsize, node_amt*batchsize]

    # print(big_batch_adjacency.shape)
    exit()

    big_batch_edges = []
    big_batch_ego_edges = []
    big_batch_positions = []
    big_batch_adjacency = torch.zeros(T, max_nodes*B, max_nodes*B)
    # edge_indicies = [batch, timestamp, [2, N]]
    for t in range(T):
        edges_at_t = []
        ego_edges_at_t = []
        positions_at_t = []
        adjacency_at_t = []
        
        for b in range(B):
            # Get the edge-index for batch element b at timestamp t.
            e = edge_indices[b][t]  # shape [2, N_b]
            ee = ego_edge_indices[b][t]
            p = positions_batch[b, t, :, :2] # shape [node_amt, 2]
            a = adjacency_batch[b, t, :, :] # [batchsize, time_stamp, node_amt, node_amt] -> [node_amt, node_amt]

            # Offset the node indices so that nodes in batch b get indices in [b*max_nodes, (b+1)*max_nodes)
            e_offset = e + b * max_nodes
            ee_offset = ee + b * max_nodes
            # p_offset = p + b*max_nodes # The positions don't need to be increased? I ws not cooking
            
            edges_at_t.append(e_offset)
            ego_edges_at_t.append(ee_offset)
            # positions_at_t.append(p_offset)
            positions_at_t.append(p)
            adjacency_at_t.append(a)

        # Concatenate all batches’ edge indices for this timestamp along the second dimension.
        combined_edges = torch.cat(edges_at_t, dim=1).to(torch.device('cuda:0'))  # shape [2, total_edges_at_t]
        big_batch_edges.append(combined_edges)

        combined_ego_edges = torch.cat(ego_edges_at_t, dim=1).to(torch.device('cuda:0'))
        big_batch_ego_edges.append(combined_ego_edges)

        combined_pos = torch.cat(positions_at_t, dim=0).to(torch.device('cuda:0'))
        big_batch_positions.append(combined_pos)

        combined_adj = torch.block_diag(adjacency_at_t)

        

    big_batch_positions = torch.stack(big_batch_positions, dim=0).to(torch.device('cuda:0'))
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
        'big_batch_ego_edges': big_batch_ego_edges,
    }

if __name__ == '__main__':
    # Create dataset
    dataset = GCNDataset('test_data_Ego.pt')

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        # print(batch['positions'].shape)
        break

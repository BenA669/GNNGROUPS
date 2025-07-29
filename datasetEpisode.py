import torch
from torch.utils.data import Dataset, DataLoader
from configReader import read_config

class oceanDataset(Dataset):
    def __init__(self, pos_path, adj_path):
        super().__init__()
        self.data_pos = torch.load(pos_path, weights_only=True)
        self.data_adj = torch.load(adj_path, weights_only=True)

    def __len__(self):
        return len(self.data_pos)
    
    def __getitem__(self, idx): 
        (positions_t_n_xy, n_hop_adjacency_t_h_n_n) = self.data_pos[idx], self.data_adj[idx]

        return (positions_t_n_xy, n_hop_adjacency_t_h_n_n)

def ocean_collate_fn(batch):
    positions_b_t_n_xy, n_hop_adjacency_b_t_h_n_n = zip(*batch)

    

class GCNDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = torch.load(data_path, weights_only=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (positions, adjacency, edge_indices, group_amt) = self.data[idx]
        # (positions, adjacency, edge_indices, _, _, _) = self.data[idx]
        # Convert everything to tensors
        ego_index, pruned_adj, reachable = getEgo(positions, adjacency, union=False, min_groups=4)
        return (positions, adjacency, edge_indices, ego_index, pruned_adj, reachable)

def collate_fn(batch):
    # Unzip the batch each sample is a tuple
    positions, adjacency, edge_indices, ego_index_batch, pruned_adj, reachable = zip(*batch)
    B = len(edge_indices)
    T = len(edge_indices[0])    

    
    # Stack the ego_positions
    positions_batch = torch.stack(positions, dim=0) # [batchsize, time_stamp, node_amt, 3]
    ego_mask_batch = torch.stack(reachable, dim=0) # [batchsize, time_stamp, node_amt]
    adjacency_batch = torch.stack(adjacency, dim=0) # [batchsize, time_stamp, node_amt, node_amt]
    pruned_adj_batch = torch.stack(pruned_adj, dim=0) # [batchsize, time_stamp, node_amt, node_amt] 

    
    # Batch to BigBatch yassification
    # [batchsize, time_stamp, node_amt, node_amt] -> [time_stamp, node_amt*batchsize, node_amt*batchsize]
    
    big_batch_positions = []
    big_batch_adjacency = []
    big_batched_adjacency_pruned = []
    for t in range(T):
        edges_at_t = []
        positions_at_t = []
        adjacency_at_t = []
        adjacency_pruned_at_t = []


        for b in range(B):
            # Get the edge-index for batch element b at timestamp t.
            # ee = ego_edge_indices[b][t]
            p = positions_batch[b, t, :, :2] # shape [node_amt, 2]
            a = adjacency_batch[b, t, :, :] # [batchsize, time_stamp, node_amt, node_amt] -> [node_amt, node_amt]
            a_p = pruned_adj_batch[b, t, :, :]
            # ego_edges_at_t.append(ee_offset)
            # positions_at_t.append(p_offset)
            positions_at_t.append(p)
            adjacency_at_t.append(a)
            adjacency_pruned_at_t.append(a_p)

        combined_pos = torch.cat(positions_at_t, dim=0).to(torch.device('cuda:0'))
        big_batch_positions.append(combined_pos)

        stacked_adj = torch.stack(adjacency_at_t, dim=0)
        combined_adj = torch.block_diag(*stacked_adj)
        big_batch_adjacency.append(combined_adj)

        stacked_adj_p = torch.stack(adjacency_pruned_at_t, dim=0)
        combined_adj_p = torch.block_diag(*stacked_adj_p)
        big_batched_adjacency_pruned.append(combined_adj_p)
        

    big_batch_positions = torch.stack(big_batch_positions, dim=0).to(torch.device('cuda:0'))
    big_batch_adjacency = torch.stack(big_batch_adjacency, dim=0).to(torch.device('cuda:0'))
    big_batched_adjacency_pruned = torch.stack(big_batched_adjacency_pruned, dim=0).to(torch.device('cuda:0'))
    return {
        'positions': positions_batch,
        'adjacency': adjacency,
        'ego_mask_batch': ego_mask_batch, # (batchsize, time_stamp, node_amt)
        'big_batch_positions': big_batch_positions, # (time_stamp, node_amt*batchsize, 2)
        'big_batch_adjacency': big_batch_adjacency, # (time_stamp, node_amt*batchsize, node_amt*batchsize)
        'big_batched_adjacency_pruned': big_batched_adjacency_pruned, # (time_stamp, node_amt*batchsize, node_amt*batchsize)
        'ego_index_batch': ego_index_batch,
    }

if __name__ == '__main__':
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    pos_path = dataset_cfg["dir_path"] + dataset_cfg["dataset_name"] + "_pos.pt"
    adj_path = dataset_cfg["dir_path"] + dataset_cfg["dataset_name"] + "_adj.pt"

    dataset = oceanDataset(pos_path, adj_path)

    batch_size = training_cfg["batch_size"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(len(batch))
        print()
    exit()

#    # Create dataset
#    dataset = GCNDataset(dataset_cfg["val_path"])
#
#    # Create DataLoader
#    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
#
#    for batch_idx, batch in enumerate(dataloader):
#        # print(batch['ego_mask_batch'].shape)
#        # print(batch)
#        for key, value in batch.items():
#            print(key)
#            print(type(value))
#            if isinstance(value, tuple):
#                print(value)
#                continue
#            print(value.shape)
#            print()
#        break
#
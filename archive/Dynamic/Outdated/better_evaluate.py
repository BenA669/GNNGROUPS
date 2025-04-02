import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from makeEpisode import makeDatasetDynamicPerlin, getEgo
from trainDyn_NodeCentric import collate_fn, infonce_loss_fn
from model import TemporalGCN

class GCNDataset(Dataset):
    def __init__(self, dataset_amt):
        super().__init__()
        self.data = []

        time_steps = 20
        group_amt = 4
        node_amt = 400

        noise_scale = 0.05      # frequency of the noise
        noise_strength = 2      # influence of the noise gradient
        tilt_strength = 0.25     # constant bias per group

        for i in range(dataset_amt):
            positions, adjacency, edge_indices = makeDatasetDynamicPerlin(
            node_amt=node_amt,
            group_amt=group_amt,
            std_dev=1,
            time_steps=time_steps,
            distance_threshold=2,
            intra_prob=0.05,
            inter_prob=0.001,
            noise_scale=noise_scale,
            noise_strength=noise_strength,
            tilt_strength=tilt_strength,
            octaves=1,
            persistence=0.5,
            lacunarity=2.0
        )
        ego_idx, ego_positions, ego_adjacency, ego_edge_indices, EgoMask = getEgo(positions, adjacency)

        self.data.append((positions, adjacency, edge_indices, ego_idx, ego_positions, ego_adjacency, ego_edge_indices, EgoMask))
        
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=int, default=1)

    args = parser.parse_args()
    eval_dataset = GCNDataset(1)

    eval_loader = DataLoader(eval_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input_dim = 2
    output_dim = 16
    num_nodes = 400 
    num_timesteps = 20 
    hidden_dim = 64 


    model = TemporalGCN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_nodes=num_nodes,
        num_timesteps=num_timesteps,
        hidden_dim=hidden_dim
    ).to(device)


    checkpoint_path = "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    model.eval()

    for batch_idx, batch in enumerate(eval_dataset):
        positions = batch['positions']
        groups = positions[:, 0, :, 2]
        ego_mask_batch = batch['ego_mask_batch']
        big_batch_edges = batch['big_batch_edges']
        big_batch_positions = batch['big_batch_positions']

        
        emb = model(big_batch_positions, big_batch_edges, ego_mask_batch)
        
        loss = infonce_loss_fn(emb, groups)
from makeEpisode import makeDatasetDynamicPerlin, getEgo
import torch
from tqdm import tqdm

if __name__ == "__main__":
    
    time_steps = 20
    group_amt = 4
    node_amt = 400
    distance_threshold = 2
    intra_prob = 0.05
    inter_prob = 0.001

    NUM_SAMPLES_TEST = 1000
    NUM_SAMPLES_VAL = 500

    test_data = []
    val_data = []

    noise_scale = 0.05      # frequency of the noise
    noise_strength = 2      # influence of the noise gradient
    tilt_strength = 0.25     # constant bias per group

    # Generate test data
    for _ in tqdm(range(NUM_SAMPLES_TEST)):
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
        test_data.append((positions, adjacency, edge_indices, ego_idx, ego_positions, ego_adjacency, ego_edge_indices, EgoMask))

    torch.save(test_data, "test_data_Ego.pt")

    # Generate validation data
    for _ in tqdm(range(NUM_SAMPLES_VAL)):
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
        val_data.append((positions, adjacency, edge_indices, ego_idx, ego_positions, ego_adjacency, ego_edge_indices, EgoMask))

    # Save datasets to file
    
    torch.save(val_data, "val_data_Ego.pt")

    print("Finished generating and saving test_data.pt and val_data.pt")
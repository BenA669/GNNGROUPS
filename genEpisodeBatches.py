from makeEpisode import makeDatasetDynamicPerlin, getEgo
import torch
from tqdm import tqdm
from configReader import read_config



if __name__ == "__main__":

    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")
    
    time_steps = dataset_cfg["timesteps"]
    group_amt = dataset_cfg["groups"]
    mixed = dataset_cfg["mixed"]
    node_amt = dataset_cfg["nodes"]

    distance_threshold = dataset_cfg["distance_threshold"]
    noise_scale = dataset_cfg["noise_scale"]      # frequency of the noise
    noise_strength = dataset_cfg["noise_strength"]      # influence of the noise gradient
    tilt_strength = dataset_cfg["tilt_strength"]     # constant bias per group
    boundary =dataset_cfg["boundary"]

    hops = dataset_cfg["hops"]

    dir_path = dataset_cfg["dir_path"]
    dataset_name = dataset_cfg["dataset_name"]

    train_name=dataset_cfg["train_path"]
    val_name=dataset_cfg["val_path"]

    samples = dataset_cfg["samples"]

    test_data = []
    val_data = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    
    def Gen(data, train=True):
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.initial_seed())
        
        for _ in tqdm(range(samples)):
            positions, adjacency, edge_indices, groups = makeDatasetDynamicPerlin(
                node_amt=node_amt,
                group_amt=group_amt,
                time_steps=time_steps,
                distance_threshold=distance_threshold,
                noise_scale=noise_scale,
                noise_strength=noise_strength,
                tilt_strength=tilt_strength,
                boundary=boundary,
                mixed=mixed,
                rng=rng
            )
            # ego_index, pruned_adj, reachable = getEgo(positions, adjacency, hop=hops, union=False, min_groups=group_amt)
            data.append((positions, adjacency, edge_indices, groups))
        if train:
            torch.save(data, train_name)
        else:
            torch.save(data, val_name)
            
    Gen(test_data, train=True)
    Gen(val_data, train=False)

print("Finished generating and saving {} and {}".format(train_name, val_name))
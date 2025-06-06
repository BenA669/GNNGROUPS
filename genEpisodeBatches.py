from makeEpisode import makeDatasetDynamicPerlin, getEgo
import torch
from tqdm import tqdm
import configparser

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')
    
    time_steps = int(config["dataset"]["timesteps"])
    group_amt = int(config["dataset"]["groups"])
    mixed = bool(config["dataset"]["mixed"])
    node_amt = int(config["dataset"]["nodes"])

    distance_threshold = int(config["dataset"]["distance_threshold"])
    noise_scale = float(config["dataset"]["noise_scale"])      # frequency of the noise
    noise_strength = float(config["dataset"]["noise_strength"])      # influence of the noise gradient
    tilt_strength = float(config["dataset"]["tilt_strength"])     # constant bias per group
    boundary = int(config["dataset"]["boundary"])

    hops = int(config["dataset"]["hops"])

    # train_name = str(config["dataset"]["dataset_train"])
    # val_name = str(config["dataset"]["dataset_val"])

    dir_path = str(config["dataset"]["dir_path"])
    dataset_name = str(config["dataset"]["dataset_name"])

    train_name="{}{}_train.pt".format(dir_path, dataset_name)
    val_name="{}{}_val.pt".format(dir_path, dataset_name)

    samples = int(config["dataset"]["samples"])

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
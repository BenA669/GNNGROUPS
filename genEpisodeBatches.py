from makeEpisode import makeDatasetDynamicPerlin, getEgo
import torch
from tqdm import tqdm
import configparser

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')
    
    time_steps = int(config["dataset"]["timesteps"])
    group_amt = int(config["dataset"]["groups"])
    node_amt = int(config["dataset"]["nodes"])

    distance_threshold = int(config["dataset"]["distance_threshold"])
    noise_scale = float(config["dataset"]["noise_scale"])      # frequency of the noise
    noise_strength = float(config["dataset"]["noise_strength"])      # influence of the noise gradient
    tilt_strength = float(config["dataset"]["tilt_strength"])     # constant bias per group
    boundary = int(config["dataset"]["boundary"])

    hops = int(config["dataset"]["hops"])
    min_groups = int(config["dataset"]["min_groups"])

    train_name = str(config["dataset"]["dataset_train"])
    val_name = str(config["dataset"]["dataset_val"])

    samples = int(config["dataset"]["samples"])

    test_data = []
    val_data = []

    
    def Gen(data, train=True):
        for _ in tqdm(range(samples)):
            positions, adjacency, edge_indices = makeDatasetDynamicPerlin(
                node_amt=node_amt,
                group_amt=group_amt,
                time_steps=time_steps,
                distance_threshold=distance_threshold,
                noise_scale=noise_scale,
                noise_strength=noise_strength,
                tilt_strength=tilt_strength,
                boundary=boundary
            )
            ego_index, pruned_adj, reachable = getEgo(positions, adjacency, hop=hops, union=False, min_groups=min_groups)
            data.append((positions, adjacency, edge_indices, ego_index, pruned_adj, reachable))
        if train:
            torch.save(data, train_name)
        else:
            torch.save(val_data, val_name)
    Gen(test_data, train=True)

    Gen(val_data, train=False)

print("Finished generating and saving {} and {}".format(train_name, val_name))
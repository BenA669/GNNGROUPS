import torch
import random
from configReader import read_config
from makeEpisode import getEgo
from animate import plot_faster

if __name__ == '__main__':
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")
    data_path = dataset_cfg["val_path"]


    data = torch.load(data_path)

    idx = random.randint(0, len(data))
    (positions, adjacency, edge_indices, group_amt) = data[idx]
    ego_index, pruned_adj, reachable = getEgo(positions, adjacency, union=False, min_groups=4)

    plot_faster(all_positions_cpu=positions,
                adjacency_dynamic_cpu=adjacency,
                ego_idx=ego_index,
                ego_mask=reachable)

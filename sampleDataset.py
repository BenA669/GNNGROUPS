import torch
import random
from configReader import read_config, getDataset, batch_to_model, genAnchors
from makeEpisode import getEgo
from animate import plot_faster
from pygameAnimate import animatev2

if __name__ == '__main__':
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    train_loader, _ = getDataset()

    for batch in train_loader:
        new_positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, anchor_indices_n = genAnchors(batch[0].squeeze())
        animatev2(new_positions_t_n_xy, A_t_n_n, anchor_indices_n)
        exit()


    # model_cfg, dataset_cfg, training_cfg = read_config("config.ini")
    # data_path = dataset_cfg["val_path"]


    # data = torch.load(data_path)

    # idx = random.randint(0, len(data))
    # (positions, adjacency, edge_indices, group_amt) = data[idx]
    # ego_index, pruned_adj, reachable = getEgo(positions, adjacency, union=False, min_groups=4)

    # plot_faster(all_positions_cpu=positions,
    #             adjacency_dynamic_cpu=adjacency,
    #             ego_idx=ego_index,
    #             ego_mask=reachable)

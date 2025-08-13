import torch
from datasetEpisode import GCNDataset, collate_fn
from tqdm import tqdm
from configReader import read_config, getDataset, getModel
from makeEpisode import genAnchors

if __name__ == "__main__":
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    train_loader, validation_loader = getDataset()
    model = getModel(eval=False)

    for batch in tqdm(train_loader):
        global_positions_t_n_xy = batch[0].squeeze()
        positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy = genAnchors(global_positions_t_n_xy)

        model_out = model(Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy)
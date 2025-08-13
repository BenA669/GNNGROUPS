import torch
from torch.utils.data.sampler import SubsetRandomSampler
import configparser
import importlib
from pathlib import Path
import numpy as np


def read_config(config_path: str = "config.ini"):
    config = configparser.ConfigParser()
    config.read(Path(config_path))

    def _parse_section(section: str) -> dict:
        out = {}
        for key, raw_val in config[section].items():
            try:
                out[key] = config.getint(section, key)
                continue
            except ValueError:
                pass

            try:
                out[key] = config.getfloat(section, key)
                continue
            except ValueError:
                pass

            try:
                out[key] = config.getboolean(section, key)
                continue
            except ValueError:
                pass

            out[key] = raw_val

        return out

    model_cfg = _parse_section("model")       
    dataset_cfg = _parse_section("dataset")   
    training_cfg = _parse_section("training") 

    dir_path     = str(dataset_cfg.get("dir_path", ""))
    dataset_name = str(dataset_cfg.get("dataset_name", ""))

    dataset_cfg["train_path"] = f"{dir_path}{dataset_name}_train.pt"
    dataset_cfg["val_path"]   = f"{dir_path}{dataset_name}_val.pt"

    dataset_cfg["adj_path"] = f"{dir_path}{dataset_name}_adj.pt"
    dataset_cfg["pos_path"] = f"{dir_path}{dataset_name}_pos.pt"

    model_cfg["config"] = config

    training_cfg["model_save"] = "{}{}".format(dir_path, training_cfg["model_name_pt"])

    return model_cfg, dataset_cfg, training_cfg


model_cfg, dataset_cfg, training_cfg = read_config("config.ini")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getModel(eval: bool=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    module = importlib.import_module("models")
    model_type = model_cfg['model_type']
    class_type = getattr(module, model_type)

    model = class_type(model_cfg["config"]).to(device)

    
    if eval:
        dir_path   = dataset_cfg.get("dir_path", "")
        model_name = training_cfg.get("model_name_pt")
        if model_name is None:
            raise KeyError("`model_name_pt` not found in [training] section of config.ini")

        checkpoint_path = f"{dir_path}{model_name}"
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    return model

def getDataset():
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    pos_path    = dataset_cfg["pos_path"]
    batch_size   = training_cfg["batch_size"]
    val_split   = training_cfg["val_split"]
    samples     = training_cfg["samples"]

    # Get Dataset type
    module = importlib.import_module("datasetEpisode") 
    dataloader_type = dataset_cfg['dataset_loader']
    class_type = getattr(module, dataloader_type)

    dataset = class_type(pos_path) # Get Dataset

    # Split indices
    indices = list(range(samples))
    split = int(np.floor(val_split * samples))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Generate Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    
    return train_loader, validation_loader

def genAnchors(positions_t_n_xy,
               anchor_ratio         = dataset_cfg['anchor_node_ratio'],
               distance_threshold   = dataset_cfg['distance_threshold']):
    time_steps, node_amt, _ = positions_t_n_xy.shape

    # Select anchor nodes
    anchor_amt = int(np.floor(node_amt * anchor_ratio))
    anchor_indices_n = torch.randperm(node_amt)[:anchor_amt]
    # print(f"anchor indices: {anchor_indices_n}")

    # Make anchors still
    new_positions_t_n_xy = positions_t_n_xy.clone()
    new_positions_t_n_xy[:, anchor_indices_n, :] = new_positions_t_n_xy[0, anchor_indices_n, :]

    # Initialize X Distance Matrix
    # Setup anchor-agent and anchor-anchor distances of X(N, N)
    X = torch.cdist(new_positions_t_n_xy, new_positions_t_n_xy)     # Get full distance matrix
    X_temp = X.clone()

    mask_anchor = torch.zeros((node_amt, node_amt), dtype=torch.bool)   # Mask out 
    mask_anchor[anchor_indices_n, :] = True
    mask_anchor[:, anchor_indices_n] = True
    X[:, ~mask_anchor] = 0

    A_t_n_n = X_temp < distance_threshold # Create Adj Matrix
    Xhat_t_n_n = A_t_n_n * X # Create Xhat

    anchor_pos_t_n_xy = torch.zeros(time_steps, node_amt, 2)
    anchor_pos_t_n_xy[:, anchor_indices_n, :] = new_positions_t_n_xy[:, anchor_indices_n, :]

    return new_positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, anchor_indices_n

def batch_to_model(batch):
    global_positions_t_n_xy = batch[0].squeeze()
    positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, _ = genAnchors(global_positions_t_n_xy)

    return (positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy)
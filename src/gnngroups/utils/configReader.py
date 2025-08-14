import sys
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model_cfg, dataset_cfg, training_cfg, device



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



def batch_to_model(batch):
    global_positions_t_n_xy = batch[0].squeeze()
    positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, _ = genAnchors(global_positions_t_n_xy)

    return (positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy)
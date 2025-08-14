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
    log_folder   = str(training_cfg.get("log_folder_path", ""))

    dataset_cfg["train_path"] = f"{dir_path}{dataset_name}_train.pt"
    dataset_cfg["val_path"]   = f"{dir_path}{dataset_name}_val.pt"

    dataset_cfg["adj_path"] = f"{dir_path}{dataset_name}_adj.pt"
    dataset_cfg["pos_path"] = f"{dir_path}{dataset_name}_pos.pt"

    modelname   = training_cfg["model_name_pt"]
    training_cfg["log_path"] = f"{log_folder}training_stats_{modelname}.json"

    model_cfg["config"] = config

    training_cfg["model_save"] = "{}{}".format(dir_path, training_cfg["model_name_pt"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model_cfg, dataset_cfg, training_cfg, device


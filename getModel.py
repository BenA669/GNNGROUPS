import torch
from configReader import read_config
import importlib

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

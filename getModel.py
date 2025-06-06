import torch
from configReader import read_config
import model as model_module  

def getModel(eval: bool=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    model_type = model_cfg.get("model_type")
    if model_type is None:
        raise KeyError("`model_type` not found in [model] section of config.ini")
    try:
        ModelClass = getattr(model_module, model_type)
    except AttributeError:
        raise ImportError(f"Model class `{model_type}` not found in model.py")

    
    model = ModelClass(model_cfg).to(device)

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

import configparser
from pathlib import Path

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

    model_cfg["config"] = config

    return model_cfg, dataset_cfg, training_cfg

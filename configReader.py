import configparser
from pathlib import Path

def read_config(config_path: str = "config.ini"):
    """
    Reads the given INI file and returns a tuple of three dicts:
      (model_cfg, dataset_cfg, training_cfg)

    - model_cfg contains everything under [model]
    - dataset_cfg contains everything under [dataset]
    - training_cfg contains everything under [training]

    Each value is converted to int/float/bool where appropriate; otherwise it's left as str.
    """
    # 1) Create parser and read the file
    config = configparser.ConfigParser()
    config.read(Path(config_path))

    def _parse_section(section: str) -> dict:
        """
        Given a section name, iterate over all (key, raw_value) pairs in that section
        and convert each raw_value to int, float, bool, or leave as str depending on its form.
        """
        out = {}
        for key, raw_val in config[section].items():
            # Try int
            try:
                out[key] = config.getint(section, key)
                continue
            except ValueError:
                pass

            # Try float
            try:
                out[key] = config.getfloat(section, key)
                continue
            except ValueError:
                pass

            # Try boolean (configparser will accept "yes"/"no", "true"/"false", etc.)
            try:
                out[key] = config.getboolean(section, key)
                continue
            except ValueError:
                pass

            # Fallback: treat as a raw string
            out[key] = raw_val

        return out

    # 2) Parse each section we care about.  If any section is missing, ConfigParser raises a KeyError.
    model_cfg = _parse_section("model")       # e.g. input_dim, hidden_dim_2, num_heads, etc. :contentReference[oaicite:0]{index=0}
    dataset_cfg = _parse_section("dataset")   # e.g. nodes, timesteps, noise_scale, mixed, dir_path, etc. :contentReference[oaicite:1]{index=1}
    training_cfg = _parse_section("training") # e.g. batch_size, epochs, learning_rate, model_name_pt, etc. :contentReference[oaicite:2]{index=2}

    dir_path     = str(dataset_cfg.get("dir_path", ""))
    dataset_name = str(dataset_cfg.get("dataset_name", ""))

    dataset_cfg["train_path"] = f"{dir_path}{dataset_name}_train.pt"
    dataset_cfg["val_path"]   = f"{dir_path}{dataset_name}_val.pt"

    model_cfg["config"] = config

    # 3) Return the three dicts
    return model_cfg, dataset_cfg, training_cfg

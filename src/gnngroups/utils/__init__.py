from .configReader import read_config

model_cfg, dataset_cfg, training_cfg, device = read_config("config.ini")

__all__ = [
    "model_cfg",
    "dataset_cfg",
    "training_cfg",
    "device",
]
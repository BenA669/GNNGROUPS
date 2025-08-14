import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import importlib
import numpy as np
from .pygameAnimate import animatev2
from .episodeOperations import genAnchors
from gnngroups.utils import *

class oceanDataset(Dataset):
    def __init__(self, pos_path):
        super().__init__()
        self.data_pos = torch.load(pos_path, weights_only=True)

    def __len__(self):
        return len(self.data_pos)
    
    def __getitem__(self, idx): 
        positions_t_n_xy  = self.data_pos[idx] 

        return positions_t_n_xy 

def getDataset():

    pos_path    = dataset_cfg["pos_path"]
    batch_size   = training_cfg["batch_size"]
    val_split   = training_cfg["val_split"]
    samples     = training_cfg["samples"]

    # Get Dataset type
    # module = importlib.import_module("datasetOperations") 
    # dataloader_type = dataset_cfg['dataset_loader']
    # class_type = getattr(module, dataloader_type)
    class_type = globals().get(dataset_cfg['dataset_loader'])
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

def sampleDataset():
    train_loader, _ = getDataset()

    for batch in train_loader:
        new_positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, anchor_indices_n = genAnchors(batch[0].squeeze())
        animatev2(new_positions_t_n_xy, A_t_n_n, anchor_indices_n)
        exit()
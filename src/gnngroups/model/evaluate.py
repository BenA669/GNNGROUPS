from torch import torch
from gnngroups.utils import *
from gnngroups.dataset import *
from .models import getModel
from .train import batch_to_modelout

# OceanGCNLSTM AvgLoss: 0.34
# OceanGCN AvgLoss: 0.39

def evaluate(display=True):
    train_loader, validation_loader     = getDataset(eval=True)
    model                               = getModel(eval=True)
    loss_func                           = torch.nn.MSELoss()

    total_loss = 0
    for batch in validation_loader:
        out_emb_t_n_o, positions_t_n_xy, A_t_n_n, anchor_indices_n = batch_to_modelout(batch, model)
        loss = loss_func(out_emb_t_n_o, positions_t_n_xy.to(device))
        total_loss += loss
        print(f"Loss: {loss}")

        if display:
            animatev2(positions_t_n_xy, A_t_n_n, anchor_indices_n, pred=out_emb_t_n_o)

            input("Press any key to continue...")

    print(f"Average Loss: {total_loss / len(validation_loader)}") 
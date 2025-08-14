import torch
import torch.optim as optim
from tqdm import tqdm
from gnngroups.utils import *
from gnngroups.dataset import *
from .models import getModel

def batch_to_modelout(batch, model):
    global_positions_t_n_xy = batch[0].squeeze()
    positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, anchor_indices_n = genAnchors(global_positions_t_n_xy)
    out_emb_t_n_o = model(Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy)

    return out_emb_t_n_o, positions_t_n_xy, A_t_n_n, anchor_indices_n

def epoch_pass(loader, model, loss_func, optimizer, train=True):
    total_loss = 0
    if train:
        desc_str = "Training"
    else:
        desc_str = "Validating"

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=desc_str):
            out_emb_t_n_o, positions_t_n_xy, _, _ = batch_to_modelout(batch, model)

            loss = loss_func(out_emb_t_n_o, positions_t_n_xy.to(device))
            total_loss += loss

            if train == True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return total_loss / len(loader)

def train():
    lr              = training_cfg["learning_rate"]
    epochs          = training_cfg["epochs"]
    model_save      = training_cfg["model_save"]

    train_loader, validation_loader     = getDataset()
    model                               = getModel(eval=False)
    print("Model is on device:", next(model.parameters()).device)
    loss_func                           = torch.nn.MSELoss()
    optimizer                           = optim.Adam(model.parameters(), lr=lr)

    hist_training_loss  = []
    hist_valid_loss     = []
    best_val_loss       = float('inf')
    for epoch in range(0, epochs):
        print(f"Epoch [{epoch}/{epochs}]:")

        train_loss = epoch_pass(train_loader, model, loss_func, optimizer)
        hist_training_loss.append(train_loss.item())
        print(f"Training Loss: {train_loss}")

        val_loss = epoch_pass(validation_loader, model, loss_func, optimizer, train=False)
        hist_valid_loss.append(val_loss.item())
        print(f"Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print("New best val loss, model saved")

        print()

    
    print(f"Training completed. Best val loss: {best_val_loss}")
    print(f"Training Loss History: ")
    print(hist_training_loss)
    print(f"Validation Loss History: ")
    print(hist_valid_loss)

if __name__ == "__main__":
    train()
import torch
import torch.optim as optim
import json
import traceback
import os
from tqdm import tqdm
from gnngroups.utils import *
from gnngroups.dataset import *
from .models import getModel
import matplotlib.pyplot as plt

def show_plot(display=True):
    log_folder_path = training_cfg['log_folder_path']
    log_path = training_cfg['log_path']
    model_type = model_cfg['model_type']
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    tloss = log_data['tloss']
    vloss = log_data['vloss']
    tgloss = log_data['tgloss']
    vgloss = log_data['vgloss']
    epochs = range(1, len(tloss) + 1) 

    plt.figure(figsize=(10, 6)) 
    plt.plot(epochs, tloss, label='Training Loss')
    plt.plot(epochs, vloss, label='Validation Loss')
    plt.plot(epochs, tgloss, label='Global Training Loss')
    plt.plot(epochs, vgloss, label='Global Validation Loss')

    plt.title(f'{model_type}: Training and Validation Loss over Epochs') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig(f"{log_folder_path}Log_{model_type}.png")
    if display:
        plt.show() 

def batch_to_modelout(batch, model):
    global_positions_t_n_xy = batch[0].squeeze()
    positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy, anchor_indices_n = genAnchors(global_positions_t_n_xy)
    out_emb_t_n_o = model(Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy)

    return out_emb_t_n_o, positions_t_n_xy, A_t_n_n, anchor_indices_n

def epoch_pass(loader, model, loss_func, optimizer, train=True, anchor_only=True):
    global_total_loss   = 0
    total_loss          = 0
    batch_amt           = len(loader)

    if train:
        desc_str = "Training"
    else:
        desc_str = "Validating"

    with torch.set_grad_enabled(train): # Disable grad when evaluating
        for batch in tqdm(loader, desc=desc_str):
            out_emb_t_n_o, positions_t_n_xy, _, anchor_indices_n = batch_to_modelout(batch, model)

            # Global position loss for data collecting
            global_loss = loss_func(out_emb_t_n_o, positions_t_n_xy.to(device))
            global_total_loss += global_loss

            if anchor_only:
                # Loss only on anchor positions
                loss = loss_func(out_emb_t_n_o[:, anchor_indices_n], positions_t_n_xy[:, anchor_indices_n].to(device))
            else:
                # Loss on all positions
                loss = loss_func(out_emb_t_n_o, positions_t_n_xy.to(device))

            total_loss += loss


            if train == True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    avg_loss        = (total_loss / batch_amt)
    avg_global_loss = (global_total_loss / batch_amt)

    return avg_loss, avg_global_loss

def train():
    lr              = training_cfg["learning_rate"]
    epochs          = training_cfg["epochs"]
    model_save      = training_cfg["model_save"]
    anchor_only     = training_cfg["anchor_only"]

    save_dir        = dataset_cfg["dir_path"] 
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving model to {model_save}")

    train_loader, validation_loader     = getDataset()
    model                               = getModel(eval=False)
    loss_func                           = torch.nn.MSELoss()
    optimizer                           = optim.Adam(model.parameters(), lr=lr)
    print("Model is on device:", next(model.parameters()).device)

    hist_training_loss         = []
    hist_valid_loss            = []
    hist_training_global_loss  = []
    hist_valid_global_loss     = []
    best_val_loss              = float('inf')
    try:
        for epoch in range(0, epochs):
            print(f"Epoch [{epoch}/{epochs}]:")

            train_loss, train_global_loss = epoch_pass(train_loader, model, loss_func, optimizer, train=True, anchor_only=anchor_only)
            hist_training_loss.append(train_loss.item())
            hist_training_global_loss.append(train_global_loss.item())
            print(f"Training Loss: {train_loss}")
            print(f"Global Training Loss: {train_global_loss}")

            val_loss, val_global_loss = epoch_pass(validation_loader, model, loss_func, optimizer, train=False, anchor_only=anchor_only)
            hist_valid_loss.append(val_loss.item())
            hist_valid_global_loss.append(val_global_loss.item())
            print(f"Validation Loss: {val_loss}")
            print(f"Global Validation Loss: {val_global_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save)
                print("New best val loss, model saved")

            print()
    except Exception as e:
        traceback.print_exc()
        print("ENDED")

    
    print(f"Training completed. Best val loss: {best_val_loss}")
    print(f"Training Loss History: ")
    print(hist_training_loss)
    print(f"Validation Loss History: ")
    print(hist_valid_loss)
    print(f"Global Training Loss History: ")
    print(hist_training_global_loss)
    print(f"Global Validation Loss History: ")
    print(hist_valid_global_loss)

    while True:
        write_log = input("Write log? y/n")
        if write_log == 'y':
            data_dict = {
                "tloss" : hist_training_loss,
                "vloss" : hist_valid_loss,
                "tgloss": hist_training_global_loss,
                "vgloss": hist_valid_global_loss,
                "epoch" : f"{epoch} / {epochs}",
                "model" : model_save,
                "mcfg"  : model_cfg,
                "dcfg"  : dataset_cfg,
                "tcfg"  : training_cfg
            }

            # data_dict.pop('config')
            data_dict["mcfg"].pop('config')
            print(data_dict)


            log_path = training_cfg['log_path']
            print(log_path)
            print(type(data_dict))
            with open(log_path, 'w') as log_file:
                json.dump(data_dict, log_file, indent=4)
            break
        elif write_log == 'n':
            break

if __name__ == "__main__":
    train()
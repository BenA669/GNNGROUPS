import torch
import torch.optim as optim
from datasetEpisode import GCNDataset, collate_fn
from tqdm import tqdm
from configReader import read_config, getDataset, getModel, device, batch_to_model
from makeEpisode import genAnchors


def epoch_pass(train_loader, model, loss_func, optimizer, train=True):
    total_loss = 0
    if train:
        desc_str = "Training"
    else:
        desc_str = "Validating"

    with torch.set_grad_enabled(train):
        for batch in tqdm(train_loader, desc=desc_str):
            positions_t_n_xy, Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy = batch_to_model(batch)
            out_emb_t_n_o = model(Xhat_t_n_n, A_t_n_n, anchor_pos_t_n_xy)

            loss = loss_func(out_emb_t_n_o, positions_t_n_xy.to(device))
            total_loss += loss

            if train == True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return total_loss / len(train_loader)

if __name__ == "__main__":
    model_cfg, dataset_cfg, training_cfg = read_config("config.ini")

    lr              = training_cfg["learning_rate"]
    epochs          = training_cfg["epochs"]
    model_save      = training_cfg["model_save"]

    train_loader, validation_loader     = getDataset()
    model                               = getModel(eval=False)
    loss_func                           = torch.nn.MSELoss()
    optimizer                           = optim.Adam(model.parameters(), lr=lr)

    hist_training_loss  = []
    hist_valid_loss     = []
    best_val_loss       = float('inf')
    for epoch in range(0, epochs):
        print(f"Epoch [{epoch}/{epochs}]:")

        train_loss = epoch_pass(train_loader, model, loss_func, optimizer)
        hist_training_loss.append(train_loss)
        print(f"Training Loss: {train_loss}")

        val_loss = epoch_pass(validation_loader, model, loss_func, optimizer, train=False)
        hist_valid_loss.append(hist_valid_loss)
        print(f"Validation Loss: {val_loss}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print("New best val loss, model saved")
    
    print(f"Training completed. Best val loss: {best_val_loss}")
    print(f"Training Loss History: ")
    print(hist_training_loss)
    print(f"Validation Loss History: ")
    print(hist_valid_loss)



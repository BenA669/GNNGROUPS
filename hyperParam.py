import optuna
import torch
import torch.optim as optim
from model import GCN
from evaluate import eval, InfoNCELoss
import torch.nn.functional as F
from tqdm import tqdm

graphs = torch.load('2_groups_100_nodes_pregenerated_graphs.pt')
graphs_validation = torch.load('2_groups_100_nodes_pregenerated_graphs_validation.pt')
    

def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: {}".format(device))

    # Suggest values for hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_dim1 = trial.suggest_int('hidden_dim1', 16, 128)
    output_dim = trial.suggest_int('output_dim', 4, 16)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_int('batch_size', 1, 10)

    # Initialize model, criterion, and optimizer
    model = GCN(2, hidden_dim1, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    epochs = 5000
    
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        
        # Load a batch of graphs
        batch_start = (epoch * batch_size) % len(graphs)
        batch_end = batch_start + batch_size
        batch_graphs = graphs[batch_start:batch_end]

        batch_graphs_on_device = [(data.to(device), adj.to(device), all_nodes.to(device), labels.to(device)) 
                          for data, adj, all_nodes, labels in batch_graphs]
        
        # Find loss for batch of graphs
        batch_loss = 0
        for data, adj, all_nodes, labels in batch_graphs_on_device:
            output = model(all_nodes.float(), adj.float())
            loss = InfoNCELoss(output, labels)
            loss.backward()
            batch_loss += loss.item()
        # Update weights
        optimizer.step()

        batch_loss /= batch_size
    
    # Evaluate the model
    accuracy = eval(model, 1000, graphs_validation)
    print(f"Accuracy: {accuracy}, Loss: {batch_loss}")
    return batch_loss

# Create a study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print(study.best_params)

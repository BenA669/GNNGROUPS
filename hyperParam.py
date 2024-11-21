import optuna
import torch
import torch.optim as optim
from model import GCN
from evaluate import eval
import torch.nn.functional as F
from tqdm import tqdm

def objective(trial):
    # Suggest values for hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_dim1 = trial.suggest_int('hidden_dim1', 16, 128)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_int('batch_size', 1, 10)
    step_size = trial.suggest_int('step_size', 1000, 10000)
    gamma = trial.suggest_loguniform('gamma', 0.01, 0.5)

    # Initialize model, criterion, and optimizer
    model = GCN(2, hidden_dim1, 2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Training loop
    epochs = 5000
    graphs = torch.load('pregenerated_graphs.pt')
    graphs_validation = torch.load('pregenerated_graphs_validation.pt')
    
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        
        # Load a batch of graphs
        batch_start = (epoch * batch_size) % len(graphs)
        batch_end = batch_start + batch_size
        batch_graphs = graphs[batch_start:batch_end]
        
        # Find loss for batch of graphs
        batch_loss = 0
        for data, adj, all_nodes, labels in batch_graphs:
            output = model(all_nodes.float(), adj.float())
            loss = F.cross_entropy(output, labels)
            loss.backward()
            batch_loss += loss.item()
        
        # Update weights
        optimizer.step()
        scheduler.step()
    
    # Evaluate the model
    accuracy, loss = eval(model, 1000, graphs_validation)
    print(f"Accuracy: {accuracy}, Loss: {loss}")
    return loss

# Create a study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print(study.best_params)

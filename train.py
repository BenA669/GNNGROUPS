from model import GCN, ClusterPredictor
from evaluate import eval, InfoNCELoss
from genGraphs import generate_and_save_graphs
import torch
import torch.optim as optim
import torch.nn.functional as F
import statistics
import os.path
from tqdm import tqdm

# if (os.path.isfile("pregenerated_graphs.pt")):
#     graphs = torch.load('pregenerated_graphs.pt')
# else:
#     print("No preGenGraphs found, generating... ")
#     generate_and_save_graphs()

# Check if GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: {}".format(device))

graphs = torch.load('300_nodes_pregenerated_graphs.pt')
graphs_validation = torch.load('300_nodes_pregenerated_graphs_validation.pt')

# Initialize model, criterion, and optimizer
lr = 0.001
weight_decay = 5e-4
input_dim = 2
hidden_dim1 = 64
outputdim = 16

model = GCN(input_dim, hidden_dim1, outputdim).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



cluster_model = ClusterPredictor(outputdim).to(device)
cluster_optimizer = optim.Adam(cluster_model.parameters(), lr=lr)

# Add a learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# Training loop
epochs = 10000
epochUpd = 1000

batch_size = 10

previous_losses  = [] 
loss_memory_size  = 50  


for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    cluster_optimizer.zero_grad()

    # Load a batch of graphs
    batch_start = (epoch * batch_size) % len(graphs)
    batch_end = batch_start + batch_size
    batch_graphs = graphs[batch_start:batch_end]

    batch_graphs_on_device = [(data.to(device), adj.to(device), all_nodes.to(device), labels.to(device)) 
                              for data, adj, all_nodes, labels in batch_graphs]

    # Accumulate batch loss
    batch_loss = 0
    cluster_loss_total = 0
    for data, adj, all_nodes, labels in batch_graphs_on_device:
        output = model(all_nodes.float(), adj.float())
        
        # Calculate InfoNCE Loss
        loss = InfoNCELoss(output, labels)
        batch_loss += loss

        # Predict number of clusters
        cluster_output = cluster_model(output)
        # Find the index of the maximum value
        _, max_index = torch.max(cluster_output, dim=0)

        # Convert the max_index to an integer
        num_clusters_pred = torch.tensor(max_index.item(), dtype=torch.float32).to(device)
        
        num_unique_labels = len(torch.unique(labels))

        # Calculate cluster prediction loss
        cluster_loss = F.mse_loss(num_clusters_pred, torch.tensor(num_unique_labels, dtype=torch.float32).to(device))
        cluster_loss_total += cluster_loss

    # Backpropagate for accumulated losses
    total_loss = batch_loss + cluster_loss_total
    total_loss.backward()

    # Update weights
    optimizer.step()
    cluster_optimizer.step()

    # Record average batch loss
    batch_loss_value = (batch_loss / batch_size).item()
    previous_losses.append(batch_loss_value)
    if len(previous_losses) > loss_memory_size:
        previous_losses.pop(0)

    if epoch % epochUpd == 0:
        print(f'Epoch {epoch}, Loss Average: {statistics.mean(previous_losses)}')
        # print("Evaluation: {}".format(eval(model, cluster_model, 1000, graphs_validation)))


torch.save(model.state_dict(), 'gcn_modelMIXED300.pth'.format(epochs, lr))
print("Model saved as 'gcn_modelMIXED300.pth'".format(epochs, lr))
torch.save(cluster_model.state_dict(), 'cluster_modelMIXED300.pth'.format(epochs, lr))
print("Model saved as 'gcn_modelMIXED300.pth'".format(epochs, lr))
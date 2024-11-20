from makeDataset import makeDataSet, plot_dataset
from model import GCN
from evaluate import eval
from genGraphs import generate_and_save_graphs
import torch
import torch.optim as optim
import torch.nn.functional as F
import statistics
import os.path


if (os.path.isfile("pregenerated_graphs.pt")):
    graphs = torch.load('pregenerated_graphs.pt')
else:
    print("No preGenGraphs found, generating... ")
    generate_and_save_graphs()


# Initialize model, criterion, and optimizer
lr = 0.00001
model = GCN(2, 32, 2)
criterion = torch.nn.CrossEntropyLoss()
def permutation_invariant_loss(output, labels):
    loss_a = F.cross_entropy(output, labels)
    loss_b = F.cross_entropy(output, 1 - labels)
    return torch.min(loss_a, loss_b)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# Training loop
epochs = 20000
epochUpd = 1000

batch_size = 10

current_patience = 0
patience = 500

max_accuracy = 90.0

previous_losses  = [] 
loss_stagnation_threshold  = 50  
timeToStag = 0
stagAvg = []
stagAvg_threshold = 15

for epoch in range(epochs):
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
        loss = permutation_invariant_loss(output, labels)
        loss.backward()
        batch_loss += loss.item()
    
    # Update weights
    optimizer.step()

    # Step the scheduler
    scheduler.step()

    # Record average batch loss
    batch_loss /= batch_size
    previous_losses.append(batch_loss)
    if len(previous_losses) > loss_stagnation_threshold:
        previous_losses.pop(0)

    if epoch % epochUpd == 0:
        print(f'Epoch {epoch}, Loss Average: {statistics.mean(previous_losses)}')
        # print("Evaluation: {}".format(eval(model, 200)))

# Add this after training the model
torch.save(model.state_dict(), 'gcn_model{}BatchLR{}SCHED.pth'.format(epochs, lr))
print("Model saved as 'gcn_model{}BatchLR{}SCHED.pth'".format(epochs, lr))
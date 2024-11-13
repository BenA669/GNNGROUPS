from makeDataset import makeDataSet, makePlot
from model import GCN
import torch
import torch.optim as optim
import torch.nn.functional as F


# Load data

data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)

# print(all_nodes)
# print(labels)

# Initialize model, criterion, and optimizer
model = GCN(2, 100, 2)
criterion = torch.nn.CrossEntropyLoss()
def permutation_invariant_loss(output, labels):
    loss_a = F.cross_entropy(output, labels)
    loss_b = F.cross_entropy(output, 1 - labels)
    print(torch.min(loss_a, loss_b))
    return torch.min(loss_a, loss_b)

def cooka(output, labels):
    # print(labels)
    # print(torch.nonzero(labels == 1))
    output1 = output[torch.nonzero(labels == 1)]
    output2 = output[torch.nonzero(labels == 0)]
    # print(output1)
    # print(output2)
    loss = (1 - abs(torch.sub(output1, output2)))**2
    loss = loss.mean()
    # print(loss)
    # print("LOSS^")
    return loss


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Training loop
epochs =   30000
epochUpd = 1000
current_patience = 0
patience = 1000
max_accuracy = 90.0

previous_losses  = [] 
loss_stagnation_threshold  = 50  

# print(model(all_nodes.float(), adj.float()))
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(all_nodes.float(), adj.float())
    # loss = criterion(output, labels)
    # loss = permutation_invariant_loss(output, labels)
    loss = cooka(output, labels)
    loss.backward()
    optimizer.step()

    previous_losses.append(loss.item())
    if len(previous_losses) > loss_stagnation_threshold:
        previous_losses.pop(0)

    _, predicted_labels = torch.max(output, 1)
    correct_predictions = max((predicted_labels == labels).sum().item(), (predicted_labels == 1 - labels).sum().item())
    total_predictions = labels.size(0)
    accuracy = (correct_predictions / total_predictions) * 100

    if epoch % epochUpd == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        # print(f'Accuracy: {accuracy:.2f}%')
        # torch.save(model.state_dict(), 'gcn_model.pth')
        # print("Model saved as 'gcn_model.pth'")

    if accuracy > max_accuracy:
        # print(f'Switching dataset because accuracy reached {accuracy:.2f}%')
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
        previous_losses = []  # Reset loss history after dataset switch

    # Check for loss stagnation
    if len(previous_losses) == loss_stagnation_threshold:
        max_loss = max(previous_losses)
        min_loss = min(previous_losses)
        if max_loss - min_loss < 1e-3:  # Change threshold as needed to define stagnation
            # print(f'Switching dataset due to loss stagnation at epoch {epoch}')
            data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
            previous_losses = []  # Reset loss history after dataset switch

    if current_patience > patience:
        # print(f"Switching dataset because patience reached")
        current_patience = 0
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
    else:
        current_patience += 1

# Add this after training the model
torch.save(model.state_dict(), 'gcn_model.pth')
print("Model saved as 'gcn_model.pth'")


# Visualization of results
_, predicted_labels = torch.max(output, 1)

# data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
model.eval()
with torch.no_grad():
    output = model(all_nodes.float(), adj.float())
    _, predicted_labels = torch.max(output, 1)

# Calculate accuracy
correct_predictions = max((predicted_labels == labels).sum().item(), (predicted_labels == 1 - labels).sum().item())
total_predictions = labels.size(0)
accuracy = (correct_predictions / total_predictions) * 100

# print(accuracy)

print(f'Accuracy: {accuracy:.2f}%')

print(predicted_labels)
print(labels)
makePlot(data, 2, adj, all_nodes)


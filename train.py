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
model = GCN(2, 32, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

# Training loop
epochs = 100000
current_patience = 0
patience = 1000
max_accuracy = 85.0

previous_losses  = [] 
loss_stagnation_threshold  = 50  

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(all_nodes.float(), adj.float())
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    previous_losses.append(loss.item())
    if len(previous_losses) > loss_stagnation_threshold:
        previous_losses.pop(0)

    _, predicted_labels = torch.max(output, 1)
    correct_predictions = (predicted_labels == labels).sum().item()
    total_predictions = labels.size(0)
    accuracy = (correct_predictions / total_predictions) * 100

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f'Accuracy: {accuracy:.2f}%')

    if accuracy > max_accuracy:
        print(f'Switching dataset because accuracy reached {accuracy:.2f}%')
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
        previous_losses = []  # Reset loss history after dataset switch

    # Check for loss stagnation
    if len(previous_losses) == loss_stagnation_threshold:
        max_loss = max(previous_losses)
        min_loss = min(previous_losses)
        if max_loss - min_loss < 1e-3:  # Change threshold as needed to define stagnation
            print(f'Switching dataset due to loss stagnation at epoch {epoch}')
            data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
            previous_losses = []  # Reset loss history after dataset switch

    if current_patience > patience:
        print(f"Switching dataset because patience reached")
        current_patience = 0
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
    else:
        current_patience += 1

# Add this after training the model
torch.save(model.state_dict(), 'gcn_model.pth')
print("Model saved as 'gcn_model.pth'")


# Visualization of results
_, predicted_labels = torch.max(output, 1)


data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
model.eval()
with torch.no_grad():
    output = model(all_nodes.float(), adj.float())
    _, predicted_labels = torch.max(output, 1)

# Calculate accuracy
correct_predictions = (predicted_labels == labels).sum().item()
total_predictions = labels.size(0)
accuracy = (correct_predictions / total_predictions) * 100

# print(accuracy)

print(f'Accuracy: {accuracy:.2f}%')

print(predicted_labels)
print(labels)
makePlot(data, 2, adj, all_nodes)


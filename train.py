from makeDataset import makeDataSet, makePlot, makePlotWithErrors
from model import GCN
from evaluate import eval
import torch
import torch.optim as optim
import torch.nn.functional as F
import statistics


# Load data
nodeAmount = 100
data, adj, all_nodes, labels = makeDataSet(groupsAmount=2, nodeAmount=nodeAmount)

# print(all_nodes)
# print(labels)

# Initialize model, criterion, and optimizer
model = GCN(2, 16, 2)
criterion = torch.nn.CrossEntropyLoss()
test = torch.zeros(nodeAmount, dtype=torch.int64)
def permutation_invariant_loss(output, labels):
    loss_a = F.cross_entropy(output, labels)
    loss_b = F.cross_entropy(output, 1 - labels)
    # print(torch.min(loss_a, loss_b))
    # print(labels)
    # print(1-labels)
    # print(predicted_labels)
    # print(1- predicted_labels)
    # print(test)

    return torch.min(loss_a, loss_b)

def cooka(output, labels):
    # print(labels)
    # print(torch.nonzero(labels == 1))
    output1 = output[torch.nonzero(labels == 1)]
    output2 = output[torch.nonzero(labels == 0)]
    # print(output1)
    # print(output2)
    calCLoss = (1 - abs(torch.sub(output1, output2)))**2
    loss = calCLoss.mean()

    # print (output)
    # print()
    # print (output1)
    # print ()
    # print (output2)
    # print()
    # print(loss)
    # print()
    loss = 2
    # print(loss)
    # print("LOSS^")
    return loss


optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

# Training loop
epochs =   6000
epochUpd = 1000
current_patience = 0
patience = 500
max_accuracy = 90.0

previous_losses  = [] 
loss_stagnation_threshold  = 50  
timeToStag = 0
stagAvg = []
stagAvg_threshold = 15
# print(model(all_nodes.float(), adj.float()))
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(all_nodes.float(), adj.float())
    # loss = criterion(output, labels)
    loss = permutation_invariant_loss(output, labels)
    # loss = cooka(output, labels)
    loss.backward()
    
    optimizer.step()

    previous_losses.append(loss.item())
    if len(previous_losses) > loss_stagnation_threshold:
        previous_losses.pop(0)

    _, predicted_labels = torch.max(output, 1)
    correct_predictions = max((predicted_labels == labels).sum().item(), (predicted_labels == 1 - labels).sum().item())
    total_predictions = labels.size(0)
    accuracy = (correct_predictions / total_predictions) * 100

    # print(accuracy)

    if epoch % epochUpd == 0:
        # print(f'Epoch {epoch}, Loss: {loss.item()}')
        # print(f'Epoch {epoch}, Loss Average: {statistics.mean(previous_losses)}')
        print(f'Epoch {epoch}, Loss Average: {statistics.mean(previous_losses)}')
        if (len(stagAvg) != 0):
            print(f'TTS Average: {statistics.mean(stagAvg)}')
        print("Evaluation: {}".format(eval(model, 100)))
        # print(stagAvg)
        # print(eval(model, 10))
        # print(f'Accuracy: {accuracy:.2f}%')
        # torch.save(model.state_dict(), 'gcn_model.pth')
        # print("Model saved as 'gcn_model.pth'")

    # if accuracy > max_accuracy:
    #     # print(f'Switching dataset because accuracy reached {accuracy:.2f}%')
    #     data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
    #     previous_losses = []  # Reset loss history after dataset switch

    # Check for loss stagnation
    if len(previous_losses) == loss_stagnation_threshold:
        max_loss = max(previous_losses)
        min_loss = min(previous_losses)
        if max_loss - min_loss < 2e-1:  # Change threshold as needed to define stagnation
            print(f'Switching dataset due to loss stagnation at epoch {epoch} with acc: {accuracy}')
            data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
            previous_losses = []  # Reset loss history after dataset switch
            stagAvg.append(timeToStag)
            if len(stagAvg) > stagAvg_threshold:
                stagAvg.pop(0)
            timeToStag = 0
            current_patience = 0
    
    timeToStag += 1
    if current_patience > patience:
        print(f"Switching dataset because patience reached")
        current_patience = 0
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
        timeToStag = 0
    else:
        current_patience += 1

# Add this after training the model
torch.save(model.state_dict(), 'gcn_model.pth')
print("Model saved as 'gcn_model.pth'")


# Visualization of results
_, predicted_labels = torch.max(output, 1)

print("Evaluation: {}".format(eval(model, 100)))

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
# makePlotWithErrors(data, 2, adj, all_nodes, labels, predicted_labels)
makePlot(data, 2, adj, all_nodes)


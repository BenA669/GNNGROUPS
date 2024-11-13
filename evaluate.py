import torch
from model import GCN
from makeDataset import makeDataSet, makePlot

# Load data


# Load the model
model = GCN(2, 100, 2)
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()
print("Model loaded successfully.")

sumA = 0
# Evaluate the model
for i in range(0, 100):
    data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
    with torch.no_grad():
        output = model(all_nodes.float(), adj.float())
        _, predicted_labels = torch.max(output, 1)

    # Calculate accuracy
    correct_predictions = max((predicted_labels == labels).sum().item(), (predicted_labels == 1 - labels).sum().item())
    total_predictions = labels.size(0)
    accuracy = (correct_predictions / total_predictions) * 100
    sumA += accuracy

print(f'Accuracy of the model: {accuracy:.2f}%')
print("Predicted Labels:", predicted_labels)
print("True Labels:", labels)

print("Performance: {}".format(sumA/100))

# Plot results
makePlot(data, 2, adj, all_nodes)

#70 ish
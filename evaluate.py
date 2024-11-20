import torch
from model import GCN
from makeDataset import makeDataSet, plot_dataset
import statistics
from tqdm import tqdm
import argparse

def eval(model, amt):
    accTotal = []
    model.eval()
    for i in tqdm(range(0, amt)):
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
        output = model(all_nodes.float(), adj.float())
        _, predicted_labels = torch.max(output, 1)

        # Calculate accuracy
        correct_predictions = max((predicted_labels == labels).sum().item(), (predicted_labels == 1 - labels).sum().item())
        total_predictions = labels.size(0)
        accuracy = (correct_predictions / total_predictions) * 100
        accTotal.append(accuracy)

    return statistics.mean(accTotal)

# Load data
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate GCN model.')
    parser.add_argument('--i', type=int, default=1000, help='Number of iterations for evaluation')
    parser.add_argument('--m', type=str, default=1000, help='Model')
    args = parser.parse_args()

    # Load the model
    model = GCN(2, 32, 2)
    model.load_state_dict(torch.load(args.m))
    # model.load_state_dict(torch.load('gcn_model.pth'))
    model.eval()
    print("Model loaded successfully.")

    sumA = 0
    # Evaluate the model
    iterations = args.i
    graphs = torch.load('pregenerated_graphs.pt')

    for i in tqdm(range(0, iterations)):
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2, nodeAmount=100)
        # data, adj, all_nodes, labels = graphs[i]
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

    print("Performance: {}".format(sumA/iterations))

    # Plot results
    if (predicted_labels == labels).sum().item() > (predicted_labels == 1 - labels).sum().item():
        plot_dataset(data, 2, adj, all_nodes, labels, predicted_labels)
    else:
        plot_dataset(data, 2, adj, all_nodes, labels, 1 - predicted_labels)
    # makePlot(data, 2, adj, all_nodes)

    #70 ish
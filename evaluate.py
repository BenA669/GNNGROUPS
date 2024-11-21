import torch
from model import GCN
from makeDataset import makeDataSet, plot_dataset
import statistics
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from genPerm import findRightPerm

# 96.3447 Acc after 10,000 iterations

def findSameGroups(label, labels):
    sameGroups = []
    negGroups = []
    for i in range(len(labels)):
        if labels[i] == label:
            sameGroups.append(i)
        else:
            negGroups.append(i)
    return sameGroups, negGroups

def InfoNCELoss(output, labels):
    # Get embeddings from output
    embeddings = output  # output is already the embeddings
    
    # Pick a random positive and negative example for each node
    loss = 0
    temperature = 0.1  # Temperature parameter for InfoNCE
    batch_size = embeddings.size(0)
    # batch_size = 5
    
    for i in range(batch_size):
        # Get same group and different group indices
        same_group, diff_group = findSameGroups(labels[i], labels)
        
        # Remove self from same group
        if i in same_group:
            same_group.remove(i)
            
        # Skip if no positive examples
        if len(same_group) == 0:
            continue
            
        # Randomly select one positive and one negative example
        pos_idx = same_group[torch.randint(0, len(same_group), (1,))]
        neg_idx = diff_group[torch.randint(0, len(diff_group), (1,))]
        
        # Get anchor, positive and negative embeddings
        anchor = embeddings[i]
        positive = embeddings[pos_idx]
        negative = embeddings[neg_idx]
        
        # Calculate similarity scores
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        neg_sim = F.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0))
        
        # InfoNCE loss calculation
        numerator = torch.exp(pos_sim / temperature)
        denominator = numerator + torch.exp(neg_sim / temperature)
        loss_i = -torch.log(numerator / denominator)
        
        loss += loss_i
        
    loss = loss / batch_size  # Average loss across batch
    return loss

def outputToLabels(output, labels):
    n_clusters = 2  # Set the number of clusters you expect
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    predicted_labels = spectral_clustering.fit_predict(output.detach().numpy())
    return findRightPerm(predicted_labels, labels)

def eval(model, amt, graphs):
    
    model.eval()
    # graphs = torch.load('pregenerated_graphs_validation.pt')

    _, _, _, labels = graphs[0]
    total_predictions = labels.size(0)
    accTotal = []
    for i in tqdm(range(0, amt)):
        # data, adj, all_nodes, labels = makeDataSet(groupsAmount=2)
        data, adj, all_nodes, labels = graphs[i]
        output = model(all_nodes.float(), adj.float())
        
        predicted_labels, correct_predictions = outputToLabels(output, labels)

        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        accTotal.append(accuracy)

    print(predicted_labels)
    print(labels)

    return statistics.mean(accTotal)

# Load data
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate GCN model.')
    parser.add_argument('--i', type=int, default=1000, help='Number of iterations for evaluation')
    parser.add_argument('--m', type=str, default=1000, help='Model')
    parser.add_argument('--c', action='store_true')
    args = parser.parse_args()

    # Load the model
    model = GCN(2, 32, 5)
    model.load_state_dict(torch.load(args.m))
    # model.load_state_dict(torch.load('gcn_model.pth'))
    model.eval()
    print("Model loaded successfully.")

    
    # Evaluate the model
    iterations = args.i

    graphs = torch.load('pregenerated_graphs_validation.pt')

    _, _, _, labels = graphs[0]
    total_predictions = labels.size(0)
    accTotal = []
    for i in tqdm(range(0, iterations)):
        data, adj, all_nodes, labels = makeDataSet(groupsAmount=2, nodeAmount=100)
        # data, adj, all_nodes, labels = graphs[i]
        with torch.no_grad():
            output = model(all_nodes.float(), adj.float())
            # _, predicted_labels = torch.max(output, 1)
            # n_clusters = 2  # Set the number of clusters you expect
            # spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=42)

            # # Fit and predict cluster labels
            # predicted_labels = spectral_clustering.fit_predict(output.detach().numpy())
            # # Find right permutation
            # predicted_labels, correct_predictions = findRightPerm(predicted_labels, labels.numpy())
            predicted_labels, correct_predictions = outputToLabels(output, labels)

        # Calculate accuracy
        total_predictions = labels.size(0)
        accuracy = (correct_predictions / total_predictions) * 100
        accTotal.append(accuracy)

        if args.c:
            print("Predicted Labels:", predicted_labels)
            print("True Labels:", labels)
            print(f'Accuracy of the model: {accuracy:.2f}%')
            plot_dataset(data, 2, adj, all_nodes, labels, predicted_labels)

    print(f'Accuracy of the model: {accuracy:.2f}%')
    print("Predicted Labels:", predicted_labels)
    print("True Labels:", labels)

    print("Performance: {}".format(statistics.mean(accTotal)))

    # Plot results
    plot_dataset(data, 2, adj, all_nodes, labels, predicted_labels)
    

    #70 ish
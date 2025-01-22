import torch
import warnings
from model import GCN, ClusterPredictor
from makeDataset import makeDataSetCUDA, plot_dataset
import statistics
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, DBSCAN
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import hdbscan
import warnings

# Ignore warnings about unconnected graphs in Spectral Clustering
warnings.filterwarnings("ignore", category=UserWarning, message=".*Graph is not fully connected.*")

# 96.3447 Acc after 10,000 iterations

def generate_swapped_sequences(original_sequence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: Identify the distinct numbers in the original sequence
    distinct_elements, _ = torch.unique(original_sequence, sorted=False, return_inverse=True)
    
    # Step 2: Generate all permutations of the distinct elements
    distinct_permutations = torch.tensor(list(permutations(distinct_elements.tolist())))
    
    # Step 3: For each permutation, replace elements in the original list
    swapped_sequences = []
    for perm in distinct_permutations:
        mapping = dict(zip(distinct_elements.tolist(), perm.tolist()))
        swapped_sequence = torch.tensor([mapping[element.item()] for element in original_sequence], device=device)
        swapped_sequences.append(swapped_sequence)
    
    return swapped_sequences


# def findRightPerm(predicted_labels, labels):
#     best_accuracy = 0.0
#     best_permutation = None

#     permutations = generate_swapped_sequences(predicted_labels)

#     for perm in permutations:
#         correct_predictions = torch.sum(perm == labels).item()
#         if correct_predictions > best_accuracy:
#             best_accuracy = correct_predictions
#             best_permutation = perm

#     return best_permutation, best_accuracy

def findRightPerm(predicted_labels, labels):
    # Ensure both predicted_labels and labels are of the same type\
    device = predicted_labels.device
    predicted_labels = predicted_labels.long()
    labels = labels.long()

    # Construct a cost matrix based on misalignment between true and predicted labels
    unique_labels = torch.unique(labels)
    cost_matrix = torch.zeros((len(unique_labels), len(unique_labels)), device=device)

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cost_matrix[i, j] = torch.sum((labels == true_label) & (predicted_labels != pred_label))

    # Solve assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
    best_permutation = torch.zeros_like(predicted_labels, device=device)

    # Remap predicted labels according to the optimal alignment
    for i, j in zip(row_ind, col_ind):
        best_permutation[predicted_labels == unique_labels[j]] = unique_labels[i]

    # Calculate the accuracy of this alignment
    correct_predictions = torch.sum(best_permutation == labels).item()
    best_accuracy = correct_predictions / len(labels)

    return best_permutation, best_accuracy

def findSameGroups(labels):
    same_groups_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_groups_mask = ~same_groups_mask
    return same_groups_mask, diff_groups_mask

def InfoNCELoss(output, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = output
    
    # Get batch size and temperature
    batch_size = embeddings.size(0)
    temperature = 0.1

    # Get masks for same and different groups
    same_group_mask, diff_group_mask = findSameGroups(labels)
    same_group_mask.fill_diagonal_(False)  # Remove self-similarity from positive mask

    # Calculate similarity matrix (batch_size x batch_size)
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

    # Extract positive similarities
    positive_similarities = similarity_matrix[same_group_mask].view(batch_size, -1)
    
    # Extract negative similarities
    negative_similarities = similarity_matrix[diff_group_mask].view(batch_size, -1)

    # Ensure there are positive examples for all anchors
    assert positive_similarities.size(1) > 0, "No positive examples found for anchors."

    # Randomly select one positive similarity per anchor
    random_pos_idx = torch.randint(0, positive_similarities.size(1), (batch_size,), device=device)
    positives = positive_similarities[torch.arange(batch_size, device=device), random_pos_idx]

    # Randomly select one negative similarity per anchor
    random_neg_idx = torch.randint(0, negative_similarities.size(1), (batch_size,), device=device)
    negatives = negative_similarities[torch.arange(batch_size, device=device), random_neg_idx]

    # InfoNCE loss calculation
    numerator = torch.exp(positives / temperature)
    denominator = numerator + torch.exp(negatives / temperature)
    loss = -torch.log(numerator / denominator)

    # Average loss over batch
    loss = loss.mean()

    return loss
def outputToLabels(output, labels):
    # n_clusters = 2  # Set the number of clusters you expect

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Assuming the output is of shape (300, 1)
    # cluster_output = modelCluster(output)  # Shape: [300, 1]

    # # Find the index of the maximum value
    # _, max_index = torch.max(cluster_output, dim=0)

    # # Convert the max_index to an integer
    # n_clusters = int(max_index.item()) + 1

    n_clusters = 2

    # print(n_clusters.shape)
    # print("{} Predicted clusters".format(n_clusters))
    # spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=42)
    # hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric='euclidean')

    # predicted_labels = torch.from_numpy(hdb.fit_predict(output.detach().cpu().numpy())).to(device=device)
    predicted_labels = torch.from_numpy(spectral_clustering.fit_predict(output.detach().cpu().numpy())).to(device=device)

    return findRightPerm(predicted_labels, labels)

def eval(model, amt, graphs):
    
    model.eval()
    # graphs = torch.load('Datasets/pregenerated_graphs_validation.pt')
    # graphs = torch.load('2_groups_100_nodes_pregenerated_graphs_validation.pt')

    _, _, _, labels = graphs[0]
    total_predictions = labels.size(0)
    accTotal = []
    for i in tqdm(range(0, amt)):
        # data, adj, all_nodes, labels = makeDataSetCUDA(groupsAmount=2)
        data, adj, all_nodes, labels = graphs[i]
        output = model(all_nodes.float(), adj.float())
        
        predicted_labels, accuracy = outputToLabels(output, labels)

        # Calculate accuracy
        accTotal.append(accuracy)

    # print(predicted_labels)
    # print(labels)

    return statistics.mean(accTotal)

# Load data
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate GCN model.')
    parser.add_argument('--i', type=int, default=1000, help='Number of iterations for evaluation')
    parser.add_argument('--m', type=str, default=1000, help='Model')
    parser.add_argument('--k', type=str, default=1000, help='Cluster')
    parser.add_argument('--c', action='store_true', help="Continuous?")
    parser.add_argument('--n', action='store_true', help="Gen New?")
    args = parser.parse_args()

    # Load the model
    model = GCN(2, 64, 16).to(device)
    # model = GCN(2, 32, 5).to(device)
    model.load_state_dict(torch.load(args.m))
    # model.load_state_dict(torch.load('gcn_model.pth'))
    model.eval()
    print("Model loaded successfully.")

    # modelCluster = ClusterPredictor(16).to(device)
    # modelCluster.load_state_dict(torch.load(args.k))

    # modelCluster.eval()

    # Evaluate the model
    iterations = args.i

    # graphs = torch.load('Datasets/3_groups_300_nodes_pregenerated_graphs_validation.pt')
    # graphs = torch.load('300_nodes_pregenerated_graphs_validation.pt')
    # graphs = torch.load('2_groups_100_nodes_pregenerated_graphs_validation.pt')
    graphs = torch.load('2_groups_200_nodes_2_NNSTD_0.8_0.1_pregenerated_graphs_validation_hard.pt')

    _, _, _, labels = graphs[0]
    total_predictions = labels.size(0)
    accTotal = []
    for i in tqdm(range(0, iterations)):
        if args.n:
            data, adj, all_nodes, labels = makeDataSetCUDA(groupsAmount=2, nodeAmount=100)
        else:
            data, adj, all_nodes, labels = graphs[i]

        with torch.no_grad():
            output = model(all_nodes.float(), adj.float())
            # _, predicted_labels = torch.max(output, 1)
            # n_clusters = 2  # Set the number of clusters you expect
            # spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=42)

            # # Fit and predict cluster labels
            # predicted_labels = spectral_clustering.fit_predict(output.detach().numpy())
            # # Find right permutation
            # predicted_labels, correct_predictions = findRightPerm(predicted_labels, labels.numpy())
            predicted_labels, accuracy = outputToLabels(output, labels)

        # Calculate accuracy
        total_predictions = labels.size(0)
        accTotal.append(accuracy)

        if args.c:
            print("Predicted Labels:", predicted_labels)
            print("True Labels:", labels)
            print(f'Accuracy of the model: {accuracy:.2f}%')
            plot_dataset(data, adj, all_nodes, labels, predicted_labels)
    
    print(f'Accuracy of the model: {accuracy:.2f}%')
    print("Predicted Labels:", predicted_labels)
    print("True Labels:", labels)

    print("Performance: {}".format(statistics.mean(accTotal)))

    # Plot results
    plot_dataset(data, adj, all_nodes, labels, predicted_labels)
    

    #70 ish
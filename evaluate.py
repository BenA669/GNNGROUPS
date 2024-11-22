import torch
from model import GCN
from makeDataset import makeDataSet, plot_dataset
import statistics
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from itertools import permutations


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


def findRightPerm(predicted_labels, labels):
    best_accuracy = 0.0
    best_permutation = None

    permutations = generate_swapped_sequences(predicted_labels)

    for perm in permutations:
        correct_predictions = torch.sum(perm == labels).item()
        if correct_predictions > best_accuracy:
            best_accuracy = correct_predictions
            best_permutation = perm

    return best_permutation, best_accuracy

def findSameGroups(labels):
    same_groups_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_groups_mask = ~same_groups_mask
    return same_groups_mask, diff_groups_mask

def InfoNCELoss(output, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = output
    
    # Get the batch size and temperature
    batch_size = embeddings.size(0)
    temperature = 0.1

    # Get masks for same and different groups
    same_group_mask, diff_group_mask = findSameGroups(labels)

    # Remove self similarity from same group mask
    same_group_mask.fill_diagonal_(False)

    # Calculate similarity matrix (batch_size x batch_size)
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

    # Select positive and negative examples for each anchor
    positives = torch.zeros(batch_size, device=device)
    negatives = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        # Get positive and negative indices
        pos_indices = torch.nonzero(same_group_mask[i]).squeeze(1)
        neg_indices = torch.nonzero(diff_group_mask[i]).squeeze(1)

        if len(pos_indices) == 0:
            continue  # Skip if no positive examples

        # Randomly select a positive and negative example
        pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,), device=device)]
        neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,), device=device)]

        # Get similarity scores for selected positive and negative
        positives[i] = similarity_matrix[i, pos_idx]
        negatives[i] = similarity_matrix[i, neg_idx]

    # InfoNCE loss calculation
    numerator = torch.exp(positives / temperature)
    denominator = numerator + torch.exp(negatives / temperature)
    loss = -torch.log(numerator / denominator)

    # Average loss over batch
    loss = loss.mean()

    return loss


def outputToLabels(output, labels):
    n_clusters = 2  # Set the number of clusters you expect
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=42)
    predicted_labels = torch.from_numpy(spectral_clustering.fit_predict(output.detach().cpu().numpy()))
    return findRightPerm(predicted_labels, labels)

def eval(model, amt, graphs):
    
    model.eval()
    # graphs = torch.load('Datasets/pregenerated_graphs_validation.pt')

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

    graphs = torch.load('Datasets/pregenerated_graphs_validation.pt')

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
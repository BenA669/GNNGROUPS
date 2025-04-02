from model import TemporalGCN
from torch import torch
from makeEpisode import makeDatasetDynamicPerlin, getEgo
from sklearn.cluster import KMeans
import numpy as np
import itertools
# from animate import plot_faster
from tqdm import tqdm
import hdbscan
from umap import UMAP
from datasetEpisode import GCNDataset, collate_fn
from torch.utils.data import DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getData():
    time_steps = 10
    group_amt = 3
    node_amt = 200
    distance_threshold = 2
    intra_prob = 0.05
    inter_prob = 0.001

    NUM_SAMPLES_TEST = 100
    NUM_SAMPLES_VAL = 100

    test_data = []
    val_data = []

    noise_scale = 0.05      # frequency of the noise
    noise_strength = 2      # influence of the noise gradient
    tilt_strength = 0.25     # constant bias per group

    positions, adjacency, edge_indices = makeDatasetDynamicPerlin(
        node_amt=node_amt,
        group_amt=group_amt,
        std_dev=1,
        time_steps=time_steps,
        distance_threshold=2,
        intra_prob=0.05,
        inter_prob=0.001,
        noise_scale=noise_scale,
        noise_strength=noise_strength,
        tilt_strength=tilt_strength,
        octaves=1,
        persistence=0.5,
        lacunarity=2.0
    )
    ego_idx, ego_positions, ego_adjacency, ego_edge_indices, ego_mask = getEgo(positions, adjacency)
    return positions.to(device), adjacency.to(device), edge_indices, ego_idx, ego_mask.to(device), ego_positions.to(device)

def cross_entropy_clustering(embeddings, n_clusters, n_iters=100):
    """
    Cluster the data using an EM algorithm on a Gaussian mixture model,
    which effectively minimizes a cross entropy between the data and the
    cluster distributions.
    
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features).
        n_clusters (int): The desired number of clusters.
        n_iters (int): Number of EM iterations.
    
    Returns:
        final_labels (np.ndarray): Cluster label for each sample.
        means (np.ndarray): Final cluster means.
        covariances (np.ndarray): Final cluster covariance matrices.
        priors (np.ndarray): Final cluster priors.
    """
    n_samples, n_features = embeddings.shape

    # ---------------------------
    # Initialization via KMeans
    # ---------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    means = kmeans.cluster_centers_
    covariances = np.zeros((n_clusters, n_features, n_features))
    priors = np.zeros(n_clusters)
    for k in range(n_clusters):
        cluster_data = embeddings[labels == k]
        # Use np.cov with rowvar=False. Add a small regularization term to avoid singular matrices.
        covariances[k] = np.cov(cluster_data, rowvar=False) + 1e-6 * np.eye(n_features)
        priors[k] = cluster_data.shape[0] / n_samples

    # ---------------------------
    # EM iterations
    # ---------------------------
    for i in range(n_iters):
        # E-step: compute responsibilities
        responsibilities = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            diff = embeddings - means[k]
            inv_cov = np.linalg.inv(covariances[k])
            det_cov = np.linalg.det(covariances[k])
            # Compute the Gaussian probability density for cluster k:
            exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
            coef = priors[k] / np.sqrt(((2 * np.pi) ** n_features) * det_cov)
            responsibilities[:, k] = coef * np.exp(exponent)
        # Normalize so that each row sums to 1
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step: update parameters using the soft assignments
        for k in range(n_clusters):
            Nk = responsibilities[:, k].sum()
            if Nk > 0:
                means[k] = (responsibilities[:, k][:, None] * embeddings).sum(axis=0) / Nk
                diff = embeddings - means[k]
                # Compute covariance: weighted outer products summed over samples
                covariances[k] = (
                    responsibilities[:, k][:, None, None]
                    * np.einsum("ni,nj->nij", diff, diff)
                ).sum(axis=0) / Nk
                # Regularize covariance to avoid numerical issues
                covariances[k] += 1e-6 * np.eye(n_features)
                priors[k] = Nk / n_samples

    # Final assignment: choose the cluster with maximum responsibility for each sample
    final_labels = responsibilities.argmax(axis=1)
    return final_labels, means, covariances, priors

def hbdscan_cluster(embeddings, min_cluster_size=2, min_samples=2):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embeddings)
    print(cluster_labels)
    # exit()
    return cluster_labels

def umap_hdbscan_cluster(embeddings, n_components=2, min_cluster_size=5, min_samples=5):
    """
    First reduce the dimensionality of the embeddings using UMAP, then apply HDBSCAN clustering.

    Args:
        embeddings (np.ndarray): High-dimensional data array of shape (n_samples, n_features).
        n_components (int): Number of dimensions for the UMAP embedding (typically 2 or 3).
        min_cluster_size (int): The minimum size of clusters for HDBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point in HDBSCAN.
    
    Returns:
        cluster_labels (np.ndarray): Cluster labels from HDBSCAN. Noise points are labeled as -1.
        embedding_umap (np.ndarray): The low-dimensional embedding of the data.
    """
    # Reduce dimensionality with UMAP
    reducer = UMAP(n_components=n_components, random_state=42)
    embedding_umap = reducer.fit_transform(embeddings)
    
    # Apply HDBSCAN clustering on the UMAP-reduced embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(embedding_umap)
    
    return cluster_labels, embedding_umap


def compute_best_accuracy(true_labels, pred_labels, n_clusters):
    """
    Given the ground truth labels and predicted labels, try all possible
    permutations of cluster label mappings to determine the highest accuracy.

    Args:
        true_labels (np.ndarray): Array of shape (n_samples,) with the true labels.
        pred_labels (np.ndarray): Array of shape (n_samples,) with the predicted cluster labels.
        n_clusters (int): Number of clusters.
        
    Returns:
        best_accuracy (float): Highest accuracy achieved with the best permutation.
        best_perm (tuple): The permutation (mapping) of predicted labels that gave the best accuracy.
    """
    best_accuracy = 0.0
    best_perm = None
    best_map = None
    # Iterate over all possible permutations of cluster indices (0,1,...,n_clusters-1)
    for perm in itertools.permutations(range(n_clusters)):
        # Create a copy of predicted labels mapped using the current permutation
        mapped_pred = np.zeros_like(pred_labels)
        for orig_label, new_label in enumerate(perm):
            mapped_pred[pred_labels == orig_label] = new_label
        # Compute accuracy as the fraction of labels that match the ground truth
        accuracy = np.mean(mapped_pred == true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_perm = perm
            best_map  = mapped_pred
    return best_accuracy, best_perm, best_map


def getModel():
    input_dim = 2
    output_dim = 8
    num_nodes = 200 
    num_timesteps = 10 
    hidden_dim = 64 


    model = TemporalGCN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_nodes=num_nodes,
        num_timesteps=num_timesteps,
        hidden_dim=hidden_dim
    ).to(device)


    checkpoint_path = "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    return model

if __name__ == "__main__":
    
    model = getModel()

    dataset = GCNDataset('val_data_Ego_2hop.pt')

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)


    acc_all = []
    accuracyHBD_all = []
    # for i in tqdm(range(500)):
    for batch_idx, batch in enumerate(dataloader):
        positions = batch['positions'][0] # batch, timestamp, node_amt, 3
        ego_mask_batch = batch['ego_mask_batch'][0]
        big_batch_positions = batch['big_batch_positions'][0]
        big_batch_adjacency = batch['big_batch_adjacency'][0]

        groups = positions[-1, :, 2]

        # emb = model(big_batch_positions, big_batch_adjacency, ego_mask_batch)
        emb = model(batch)
        


        # positions, adjacency, edge_indices, ego_idx, ego_mask, ego_positions = getData()
        # emb = model(positions[:, :, :2], adjacency, ego_mask, eval=True)
        # print(emb.shape)
        
        emb_np = emb.cpu().detach().numpy().squeeze(0)
        
        # emb_np = emb_np[ego_mask.cpu()[-1]]
        emb_np = emb_np[ego_mask_batch.any(dim=0).cpu()]
        # print(f"ego: {ego_mask.shape}")
        # print(f"union ego: {ego_mask.any(dim=0)}")
        # print(f"shape: {ego_mask.any(dim=0).shape}")
        # group_ids = positions[-1, ego_mask.cpu()[-1], 2].long()
        # print(f"position shape: {positions.shape}")
        group_ids = positions[-1, ego_mask_batch.any(dim=0).cpu(), 2].long()
        n_clusters = torch.unique(group_ids).size(0)
        if n_clusters == 1:
            continue
        print(f"# CLUSTERS: {n_clusters}")

        labels, means, covs, priors = cross_entropy_clustering(emb_np, n_clusters=n_clusters, n_iters=100)

        # labels = hbdscan_cluster(emb_np)
        labelsHBD, embedding_umap = umap_hdbscan_cluster(emb_np, n_components=2, min_cluster_size=2, min_samples=2)


        true_labels = group_ids.cpu().numpy()
        # print(f"true labels: {true_labels}")
        # print(f"gussed labels: {labels}")

        accuracy, best_perm, pred_groups = compute_best_accuracy(true_labels, labels, 3)

        # print("Best permutation mapping (predicted label -> true label):", best_perm)
        print("Best clustering accuracy: {:.4f}".format(accuracy))
        # embed=emb[0, ego_mask.cpu()[-1]],
        # pred_groups=pred_groups
        # plot_faster(positions.cpu(), adjacency.cpu(),  ego_idx=ego_idx, ego_network_indices=ego_positions.cpu(), embed=emb_np, ego_mask=ego_mask)
        # input("Press Enter to continue...")
        
        accuracyHBD, best_perm, pred_groups = compute_best_accuracy(true_labels, labelsHBD, 3)

        # print("Best permutation mapping (predicted label -> true label):", best_perm)
        print("Best clustering accuracyHBD: {:.4f}".format(accuracyHBD))

        accuracyHBD_all.append(accuracyHBD)
        acc_avgHBD = sum(accuracyHBD_all) / len(accuracyHBD_all)
        

        acc_all.append(accuracy)
        acc_avg = sum(acc_all) / len(acc_all)
        print("ACC AVERAGE: ")
        print(acc_avg)

        print("ACCHBAD AVERAGE: ")
        print(acc_avgHBD)

    acc_avg = sum(acc_all) / len(acc_all)
    print("ACC AVERAGE: ")
    print(acc_avg)

# Acc no padd 0.76
# Acc w padd bad :(


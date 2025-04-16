from model import TemporalGCN
import torch
from makeEpisode import makeDatasetDynamicPerlin, getEgo
from sklearn.cluster import KMeans
import numpy as np
import itertools
from animate import plot_faster
from tqdm import tqdm
import hdbscan
from umap import UMAP
from datasetEpisode import GCNDataset, collate_fn
from torch.utils.data import DataLoader
import configparser
import matplotlib.pyplot as plt
import networkx as nx  # NEW IMPORT

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

    noise_scale = 0.05      # frequency of the noise
    noise_strength = 2      # influence of the noise gradient
    tilt_strength = 0.25    # constant bias per group

    positions, adjacency, edge_indices = makeDatasetDynamicPerlin(
        node_amt=node_amt,
        group_amt=group_amt,
        std_dev=1,
        time_steps=time_steps,
        distance_threshold=distance_threshold,
        intra_prob=intra_prob,
        inter_prob=inter_prob,
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
    """
    n_samples, n_features = embeddings.shape

    # Initialization via KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    means = kmeans.cluster_centers_
    covariances = np.zeros((n_clusters, n_features, n_features))
    priors = np.zeros(n_clusters)
    for k in range(n_clusters):
        cluster_data = embeddings[labels == k]
        covariances[k] = np.cov(cluster_data, rowvar=False) + 1e-6 * np.eye(n_features)
        priors[k] = cluster_data.shape[0] / n_samples

    # EM iterations
    for i in range(n_iters):
        responsibilities = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            diff = embeddings - means[k]
            inv_cov = np.linalg.inv(covariances[k])
            det_cov = np.linalg.det(covariances[k])
            exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
            coef = priors[k] / np.sqrt(((2 * np.pi) ** n_features) * det_cov)
            responsibilities[:, k] = coef * np.exp(exponent)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        for k in range(n_clusters):
            Nk = responsibilities[:, k].sum()
            if Nk > 0:
                means[k] = (responsibilities[:, k][:, None] * embeddings).sum(axis=0) / Nk
                diff = embeddings - means[k]
                covariances[k] = (
                    responsibilities[:, k][:, None, None]
                    * np.einsum("ni,nj->nij", diff, diff)
                ).sum(axis=0) / Nk
                covariances[k] += 1e-6 * np.eye(n_features)
                priors[k] = Nk / n_samples

    final_labels = responsibilities.argmax(axis=1)
    return final_labels, means, covariances, priors

def hbdscan_cluster(embeddings, min_cluster_size=2, min_samples=2):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embeddings)
    print(cluster_labels)
    return cluster_labels

def umap_hdbscan_cluster(embeddings, n_components=2, min_cluster_size=5, min_samples=5):
    reducer = UMAP(n_components=n_components, random_state=42)
    embedding_umap = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(embedding_umap)
    return cluster_labels, embedding_umap

def compute_best_accuracy(true_labels, pred_labels, n_clusters):
    best_accuracy = 0.0
    best_perm = None
    best_map = None
    for perm in itertools.permutations(range(n_clusters)):
        mapped_pred = np.zeros_like(pred_labels)
        for orig_label, new_label in enumerate(perm):
            mapped_pred[pred_labels == orig_label] = new_label
        accuracy = np.mean(mapped_pred == true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_perm = perm
            best_map = mapped_pred
    return best_accuracy, best_perm, best_map

def getModel(config):
    model_name = config["training"]["model_name"]
    model = TemporalGCN(config).to(device)
    checkpoint_path = model_name
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    groups_amt = int(config["dataset"]["groups"])
    model = getModel(config)

    dataset = GCNDataset(str(config["dataset"]["dataset_val"]))
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    # Lists to store overall metrics from each sample
    acc_all = []             # clustering accuracy from cross-entropy clustering
    accuracyHBD_all = []     # clustering accuracy from UMAP + HDBSCAN method
    ego_degrees_all = []     # number of connections (degree) for the ego node
    ego_nodes_all = []       # number of nodes in the ego network
    ego_positions_all = []   # (optional) ego node position at the final timestep
    ego_edge_connectivity_all = []  # edge connectivity within the ego network
    ego_edges_all = []       # NEW: total number of edges in the ego network

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        positions = batch['positions'][0]  # shape: [timestamp, node_amt, 3]
        adjacency = batch['adjacency'][0]
        ego_mask_batch = batch['ego_mask_batch'][0]
        big_batch_positions = batch['big_batch_positions'][0]
        big_batch_adjacency = batch['big_batch_adjacency'][0]
        ego_index_batch = batch['ego_index_batch'][0]

        emb = model(batch)
        # Assuming emb is of shape [batch, timesteps, nodes, features]
        emb_np = emb.cpu().detach().numpy().squeeze(0)[:, -1, :]
        # Use the ego mask to select nodes associated to the ego network
        selected_indices = ego_mask_batch.any(dim=0).cpu()
        emb_np = emb_np[selected_indices]

        # Extract ground truth group labels from the last timestep (assuming the third coordinate holds group id)
        true_labels = positions[-1, selected_indices, 2].long().cpu().numpy()
        n_clusters = np.unique(true_labels).size
        if n_clusters == 1:
            continue
        print(f"# CLUSTERS: {n_clusters}")

        # UMAP + HDBSCAN clustering
        labelsHBD, embedding_umap = umap_hdbscan_cluster(emb_np, n_components=2, min_cluster_size=2, min_samples=2)
        accuracyHBD, best_perm, pred_groups_hbd = compute_best_accuracy(true_labels, labelsHBD, groups_amt)

        print(f"true labels:    {true_labels}")
        print("Best clustering accuracy (HBD): {:.4f}".format(accuracyHBD))

        # Compute additional ego network features:
        # 1. Number of nodes in the ego network
        num_ego_nodes = int(ego_mask_batch.any(dim=0).sum().item())
        # 2. Ego node degree: Sum the connections of the ego node using the big batch adjacency matrix.
        #    (Assuming that big_batch_adjacency is structured as an adjacency matrix, and ego_index_batch is the row index for the ego node.)
        ego_degree = int(big_batch_adjacency[ego_index_batch].sum().item())
        # 3. (Optional) Ego node spatial position at the final time-step:
        ego_pos = positions[-1, ego_index_batch].cpu().numpy()

        # 4. Compute edge connectivity within the ego network
        ego_nodes_mask = ego_mask_batch.any(dim=0)
        ego_adj_sub = big_batch_adjacency[ego_nodes_mask][:, ego_nodes_mask]
        # Convert to NumPy (if it’s a torch tensor)
        if hasattr(ego_adj_sub, 'cpu'):
            ego_adj_sub_np = ego_adj_sub.cpu().numpy()
        else:
            ego_adj_sub_np = ego_adj_sub
        # Build an undirected graph from the adjacency matrix
        G_ego = nx.from_numpy_array(ego_adj_sub_np)
        # Compute the edge connectivity of the ego subgraph (minimum number of edges whose removal disconnects the graph)
        try:
            edge_conn = nx.edge_connectivity(G_ego)
        except Exception as e:
            edge_conn = 0
        ego_edge_connectivity_all.append(edge_conn)

        # NEW: Compute the total number of edges in the ego network
        num_edges = G_ego.number_of_edges()
        ego_edges_all.append(num_edges)

        # Save the metrics
        accuracyHBD_all.append(accuracyHBD)
        ego_degrees_all.append(ego_degree)
        ego_nodes_all.append(num_ego_nodes)
        ego_positions_all.append(ego_pos)

        # Optional: Plot the current sample (and check clustering)
        # plot_faster(positions.cpu(), adjacency.cpu(),
        #             pred_groups=pred_groups, ego_mask=ego_mask_batch, embed=emb_np)
        # input("Press Enter to continue...")

    # Print overall average accuracy
    avg_acc_hbd = sum(accuracyHBD_all) / len(accuracyHBD_all)
    print("Overall HDBSCAN Clustering ACC AVERAGE:", avg_acc_hbd)


    plt.figure()
    plt.scatter(ego_degrees_all, accuracyHBD_all, c='g')
    plt.xlabel("Ego Node Degree (Number of Connections)")
    plt.ylabel("Clustering Accuracy (HDBSCAN)")
    plt.title("HDBSCAN Accuracy vs Ego Node Degree")
    plt.grid(True)
    plt.savefig("HDBSCAN Accuracy vs Ego Node Degree.png")
    plt.show()

    plt.figure()
    plt.scatter(ego_nodes_all, accuracyHBD_all, c='m')
    plt.xlabel("Number of Nodes in Ego Network")
    plt.ylabel("HDBSCAN Accuracy (HDBSCAN)")
    plt.title("HDBSCAN Accuracy vs Ego Network Size")
    plt.grid(True)
    plt.savefig("HDBSCAN Accuracy vs Ego Network Size.png")
    plt.show()

    plt.figure()
    plt.scatter(ego_edge_connectivity_all, accuracyHBD_all, c='c')
    plt.xlabel("Ego Network Edge Connectivity")
    plt.ylabel("HDBSCAN Accuracy (HDBSCAN)")
    plt.title("HDBSCAN Accuracy vs Ego Network Edge Connectivity")
    plt.grid(True)
    plt.savefig("HDBSCAN Accuracy vs Ego Network Edge Connectivity.png")
    plt.show()

    # NEW: Plotting the number of edges in the ego network vs. HDBSCAN accuracy
    plt.figure()
    plt.scatter(ego_edges_all, accuracyHBD_all, c='orange')
    plt.xlabel("Number of Edges in Ego Network")
    plt.ylabel("HDBSCAN Accuracy (HDBSCAN)")
    plt.title("HDBSCAN Accuracy vs Number of Edges in Ego Network")
    plt.grid(True)
    plt.savefig("HDBSCAN Accuracy vs Number of Edges in Ego Network.png")
    plt.show()

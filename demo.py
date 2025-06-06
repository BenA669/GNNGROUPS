# evaluate.py
from model import *
from makeEpisode import makeDatasetDynamicPerlin, getEgo
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import configparser
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getData():
    time_steps    = 10
    group_amt     = 3
    node_amt      = 200
    distance_threshold = 2

    # Generate one “episode” (positions, adjacency, etc.)
    positions, adjacency, edge_indices, group_amt = makeDatasetDynamicPerlin(
        node_amt=node_amt,
        group_amt=group_amt,
        std_dev=1,
        time_steps=time_steps,
        distance_threshold=distance_threshold,
        noise_scale=0.05,
        noise_strength=2.0,
        tilt_strength=0.25,
        octaves=1,
        persistence=0.5,
        lacunarity=2.0
    )
    # Choose an ego node and build its ego‐network
    ego_idx, ego_positions, ego_adjacency, ego_edge_indices, ego_mask = getEgo(positions, adjacency)
    # Move to GPU/CPU as needed
    return (
        positions.to(device),            # [T, node_amt, 3]  (x, y, true_group)
        adjacency.to(device),            # [T, node_amt, node_amt]
        edge_indices,                    # list of length T (not used for plotting here)
        ego_idx,
        ego_mask.to(device),             # [T, node_amt] boolean mask of which nodes are in ego net at each t
        ego_positions.to(device)         # [T, n_ego, 3]    (x,y,true_group) restricted to ego subgraph
    )

def getModel(config):
    dir_path   = str(config["dataset"]["dir_path"])
    model_name = str(config["training"]["model_name_pt"])
    model_save = f"{dir_path}{model_name}"

    model = AttentionGCNOld(config).to(device)
    model.load_state_dict(torch.load(model_save, map_location=device))
    model.eval()
    return model

def animate_model_clustering(
    positions_full,   # Tensor[T, node_amt, 3]   (x, y, true_group)
    ego_mask,         # Tensor[T, node_amt]       boolean
    embeddings,       # NumPy array of shape [node_amt, T, D]
    n_clusters,       # int, number of true groups
    interval=500      # ms between frames
):
    """
    Build a Matplotlib FuncAnimation that, for each time‐step t:
     - Extracts the T‐step-t embedding of every active node
     - Runs KMeans(n_clusters) on those D‐dim embeddings
     - Plots the (x,y) positions (true positions) of active nodes,
       coloring each by its predicted cluster label at time t.
    """
    T, N, _ = positions_full.shape  # Actually positions_full is [T, node_amt, 3] in torch.Tensor form

    # Convert everything to CPU NumPy for faster plotting:
    pos_np      = positions_full.cpu().numpy()    # shape: [T, node_amt, 3]
    ego_mask_np = ego_mask.cpu().numpy()          # [T, node_amt] boolean
    emb_np      = embeddings                      # [node_amt, T, D], already CPU NumPy

    # Prepare figure:
    fig, ax = plt.subplots(figsize=(6,6))
    scatter = ax.scatter([], [], s=30, cmap="tab10", vmin=0, vmax=n_clusters-1)
    title   = ax.text(0.5, 1.03, "", ha="center", transform=ax.transAxes, fontsize=12)

    ax.set_xlim(pos_np[:,:,0].min() - 1, pos_np[:,:,0].max() + 1)
    ax.set_ylim(pos_np[:,:,1].min() - 1, pos_np[:,:,1].max() + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def init():
        scatter.set_offsets(np.zeros((0,2)))
        scatter.set_array(np.zeros((0,)))
        title.set_text("")
        return scatter, title

    def update(frame):
        """
        frame ranges from 0 to T-1.
        We will:
         - gather all active indices at time=frame
         - get those nodes’ 2D positions
         - get their D-dim embeddings at that time
         - run KMeans(n_clusters) → predicted labels
         - update scatter offsets + colors
        """
        t = frame
        mask_t       = ego_mask_np[t]           # boolean array of length node_amt
        active_idx   = np.nonzero(mask_t)[0]    # array of indices
        pts          = pos_np[t, active_idx, :2]  # shape: [n_active, 2]
        emb_t        = emb_np[active_idx, t, :]    # [n_active, D]

        # Run KMeans on emb_t:
        if emb_t.shape[0] >= n_clusters:
            # Only do KMeans if at least n_clusters points exist
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(emb_t)
            pred_labels = kmeans.labels_
        else:
            # fewer than n_clusters active nodes? Then just label them all 0
            pred_labels = np.zeros(emb_t.shape[0], dtype=int)

        # Update scatter:
        scatter.set_offsets(pts)
        scatter.set_array(pred_labels)  # color by cluster label
        title.set_text(f"Time-step {t}")
        return scatter, title

    anim = FuncAnimation(
        fig,
        update,
        frames=list(range(T)),
        init_func=init,
        interval=interval,
        blit=True
    )
    plt.show()


if __name__ == "__main__":
    # 1) Load config and model
    config = configparser.ConfigParser()
    config.read('config.ini')
    model = getModel(config)

    # 2) Grab one episode (positions, ego‐mask, etc)
    (
        positions_full,   # [T, node_amt, 3]
        adjacency_full,   # [T, node_amt, node_amt]  (unused in animation)
        edge_indices,     # list length T
        ego_idx,
        ego_mask,         # [T, node_amt]
        ego_positions     # [T, n_ego, 3]   (unused here—we’ll visualize the full‐graph positions)
    ) = getData()

    # 3) Build a “fake” batch dict just as evaluate.py expects:
    #    - big_batch_positions: Tensor [T, B*N, 3]   (B=1 here)
    #    - big_batched_adjacency_pruned: Tensor [T, B*N, B*N]
    #    - ego_mask_batch: Tensor [B, T, N]
    #
    #    We only need big_batch_positions and ego_mask_batch for the forward‐pass, because
    #    AttentionGCNOld only uses positions & adjacency to produce embeddings.
    #
    #    We already have `positions_full` = [T, node_amt, 3].
    #    Let B=1, so reshape to [T, 1*node_amt, 3].
    #

    T, N, _ = positions_full.shape
    B = 1
    # Create a dummy “adjacency” of zeros (AttentionGCNOld code needs a big_batched_adjacency_pruned tensor,
    # but we can just give it a zero‐matrix if we don’t care about edges for this demo.)
    dummy_adj = torch.zeros((T, B*N, B*N), device=device)

    batch = {
        'big_batch_positions':    positions_full.view(T, B*N, 3),
        'big_batched_adjacency_pruned': dummy_adj,
        'ego_mask_batch':         ego_mask.unsqueeze(0)  # [1, T, N]
    }

    # 4) Do a forward‐pass through the model to get per-node, per‐time embeddings.
    #    AttentionGCNOld’s forward returns shape [B, N, T, H].
    #
    with torch.no_grad():
        output = model(batch, eval=True)  # [1, N, T, H]
    emb = output.cpu().squeeze(0).numpy()  # now shape [N, T, H] in NumPy

    # 5) We know the “true” number of groups = the max value in positions_full[:,:,2] + 1
    true_groups = positions_full[:,:,2].long()  # Tensor [T, N]
    n_clusters  = int(true_groups.unique().size(0))

    # 6) Launch the live demo animation
    animate_model_clustering(
        positions_full,  # [T, N, 3], CPU Tensor
        ego_mask,        # [T, N], CPU Boolean Tensor
        emb,             # NumPy [N, T, H]
        n_clusters=n_clusters,
        interval=500     # 500 ms between frames; adjust as desired
    )

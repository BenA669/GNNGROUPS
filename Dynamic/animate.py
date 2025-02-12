import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from itertools import permutations

def plot_faster(all_positions_cpu, adjacency_dynamic_cpu, embed=None, 
                ego_idx=None, ego_network_indices=None, pred_groups=None, save_path="animation.gif"):
    """
    Animate the dynamic graph. If embedding is provided, a t-SNE plot is saved.
    If ego information is provided (ego_idx and ego_network_indices), nodes (and edges)
    in the ego network appear at normal opacity while all other nodes/edges are drawn
    more transparently. The ego node is further highlighted.
    
    Parameters:
      all_positions_cpu: Tensor of shape (time_steps, node_amt, 3) with node positions and group id.
      adjacency_dynamic_cpu: Tensor of shape (time_steps, node_amt, node_amt) with adjacency info.
      embed (optional): An embedding tensor for t-SNE and clustering visualization.
      ego_idx (optional): An integer giving the index of the ego node.
      ego_network_indices (optional): Either:
            - a 1D tensor or list of node indices (e.g. [3, 7, 10, ...]) that are part of the ego network, or
            - a 3D tensor (time_steps, n_ego, 3) of ego positions (from which indices will be recovered)
      save_path: Where to save the animated GIF.
    """
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embed is not None:
        n_clusters = 4
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, 
                                                 affinity='nearest_neighbors', 
                                                 n_neighbors=2, random_state=42)
        predicted_labels = torch.from_numpy(
            spectral_clustering.fit_predict(embed.detach().cpu().numpy()[0])
        ).to(device=device)
        if ego_idx is not None:
            actual_labels = ego_network_indices[0, :, 2].long().to(device)
        else:
            actual_labels = all_positions_cpu[0, :, 2].long().to(device)
        unique_actual = torch.unique(actual_labels).tolist()

        unique_pred = torch.unique(predicted_labels).tolist()

        if len(unique_actual) == len(unique_pred):
            best_accuracy = 0.0
            best_perm_map = None

            for perm in permutations(unique_pred):
                perm_map = {}
                for i, pred_id in enumerate(perm):
                    perm_map[pred_id] = unique_actual[i]

                mapped = torch.tensor([perm_map[int(lbl.item())] for lbl in predicted_labels],
                                      device=device)
                correct = (mapped == actual_labels).sum().item()
                total = actual_labels.shape[0]
                accuracy = correct / total

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_perm_map = perm_map
            
            if best_perm_map is not None:
                best_mapped_labels = torch.tensor([best_perm_map[int(lbl.item())]
                                                   for lbl in predicted_labels],
                                                  device=device)
                print(f"Accuracy found = {best_accuracy:.3f}")
        else:
            print("Warning: Number of unique actual labels != number of predicted clusters. "
                  "Skipping label permutation matching.")
            best_mapped_labels = predicted_labels

        tsneEmbed = TSNE(perplexity=5).fit_transform(embed.cpu().detach().numpy()[0])
        figEmbed = plt.figure(figsize=(10, 10)) 
        plt.title("Model Output Embeddings Visualized")
        if ego_idx is not None:
            for node in range(ego_network_indices.size(dim=1)):
                group = int(ego_network_indices[0, node, 2].item())
                plt.scatter(tsneEmbed[node, 0], tsneEmbed[node, 1], 
                            c=colors[group % len(colors)], alpha=0.8)
        else:
            for node in range(all_positions_cpu.size(dim=1)):
                group = int(all_positions_cpu[0, node, 2].item())
                plt.scatter(tsneEmbed[node, 0], tsneEmbed[node, 1], 
                            c=colors[group % len(colors)], alpha=0.8)
        plt.savefig("tsne_embeddings.png")
        plt.close()

    time_steps, node_amt, _ = all_positions_cpu.shape

    # Process the ego network information:
    if ego_network_indices is not None:
        # If ego_network_indices is a tensor and appears to be a 3D tensor (e.g. ego positions),
        # then recover the original node indices by matching rows from time step 0.
        if isinstance(ego_network_indices, torch.Tensor):
            if ego_network_indices.ndim == 3:
                candidate = ego_network_indices[0]  # shape: (n_ego, 3)
                full = all_positions_cpu[0]          # shape: (node_amt, 3)
                computed_indices = []
                for row in candidate:
                    # Check for exact equality across the 3 coordinates.
                    match = (full == row).all(dim=1)
                    indices = torch.nonzero(match, as_tuple=False).flatten().tolist()
                    if indices:
                        computed_indices.append(indices[0])
                ego_network_indices = computed_indices
            else:
                ego_network_indices = ego_network_indices.tolist()
        elif isinstance(ego_network_indices, list):
            # In case it's a list of lists (which would be unhashable), try to convert inner lists to ints.
            if len(ego_network_indices) > 0 and isinstance(ego_network_indices[0], list):
                # You might need to adjust this depending on your data.
                ego_network_indices = [int(x[0]) for x in ego_network_indices]
        # Now create a set for fast membership testing.
        ego_network_set = set(ego_network_indices)
    else:
        ego_network_set = None

    fig, ax = plt.subplots()

    def update(t):
        ax.clear()
        adj_t = adjacency_dynamic_cpu[t]  # (node_amt, node_amt)

        # --- Plot edges ---
        for i in range(node_amt):
            for j in range(i + 1, node_amt):
                if adj_t[i, j] == 1:
                    # Adjust edge opacity based on ego network membership if provided.
                    if ego_network_set is not None:
                        if (i in ego_network_set) and (j in ego_network_set):
                            edge_alpha = 0.6
                        else:
                            edge_alpha = 0.1
                    else:
                        edge_alpha = 0.2
                    x_vals = [all_positions_cpu[t, i, 0].item(), all_positions_cpu[t, j, 0].item()]
                    y_vals = [all_positions_cpu[t, i, 1].item(), all_positions_cpu[t, j, 1].item()]
                    ax.plot(x_vals, y_vals, c="gray", alpha=edge_alpha, linewidth=0.5)

        # --- Plot nodes ---
        group_ids = all_positions_cpu[t, :, 2].long()
        unique_groups = torch.unique(group_ids)

        if ego_network_set is not None:
            for g in unique_groups:
                group_mask = (group_ids == g)
                indices = torch.nonzero(group_mask, as_tuple=False).flatten().tolist()
                # Split indices into those that are in the ego network and those that are not.
                indices_in_ego = [i for i in indices if i in ego_network_set]
                indices_not_in_ego = [i for i in indices if i not in ego_network_set]
                if indices_in_ego:
                    ax.scatter(
                        all_positions_cpu[t, indices_in_ego, 0],
                        all_positions_cpu[t, indices_in_ego, 1],
                        c=colors[int(g) % len(colors)],
                        label=f"Group {g}",
                        alpha=0.7,
                    )
                if indices_not_in_ego:
                    ax.scatter(
                        all_positions_cpu[t, indices_not_in_ego, 0],
                        all_positions_cpu[t, indices_not_in_ego, 1],
                        c=colors[int(g) % len(colors)],
                        alpha=0.1,
                    )
        else:
            for g in unique_groups:
                mask = (group_ids == g)
                ax.scatter(
                    all_positions_cpu[t, mask, 0],
                    all_positions_cpu[t, mask, 1],
                    c=colors[int(g) % len(colors)],
                    label=f"Group {g}",
                    alpha=0.7,
                )

        # --- Highlight the ego node if provided ---
        if ego_idx is not None and 0 <= ego_idx < node_amt:
            ego_x = all_positions_cpu[t, ego_idx, 0].item()
            ego_y = all_positions_cpu[t, ego_idx, 1].item()
            ax.scatter(ego_x, ego_y, s=200, marker='*', c=colors[int(all_positions_cpu[t, ego_idx, 2].item()) % len(colors)], zorder=10, alpha=0.5)
            ax.text(ego_x, ego_y, "Anchor", fontsize=12, color='black', zorder=11,
                    verticalalignment='bottom', horizontalalignment='right')
            
        # print(ego_network_indices)
        # Show incorrect predictions
        if pred_groups is not None:
            i = 0
            for node in ego_network_indices:
                group = int(all_positions_cpu[0, node, 2].item())
                if pred_groups[i] != group:
                    ax.scatter(
                    all_positions_cpu[t, node, 0],
                    all_positions_cpu[t, node, 1],
                    c='red',
                    alpha=0.2,
                    s=100
                )
                
                i += 1


        ax.set_title(f"Time Step {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=50, repeat=True)
    ani.save(save_path, writer=animation.PillowWriter(fps=20))
    print(f"Animation saved as {save_path}")

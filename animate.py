import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np
from sklearn.cluster import SpectralClustering
from itertools import permutations
from sklearn.manifold import TSNE


def plot_faster(all_positions_cpu, adjacency_dynamic_cpu, embed=None, 
                ego_idx=None, ego_network_indices=None, pred_groups=None, ego_mask=None, save_path="animation.gif"):
    

    if embed is not None:
        tsne = TSNE(n_components=2, random_state=0)
        X_embedded = tsne.fit_transform(embed)

        # Plot the embedded data
        plt.clf()
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
        plt.title("t-SNE visualization of embedding")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig("t-SNE-embed.png")
        plt.clf()
    
    colors = ["red", "green", "blue", "orange", "purple", "c"]
    time_steps, node_amt, _ = all_positions_cpu.shape

    # Compute min and max for X and Y to fix axis limits
    positions_np = all_positions_cpu[:, :, :2].cpu().numpy()
    x_min, y_min = positions_np.min(axis=(0, 1))
    x_max, y_max = positions_np.max(axis=(0, 1))

    # Add padding to make it look nicer
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad


    fig, ax = plt.subplots()

    def update(t):
        ax.clear()

        # --- Vectorized edge plotting ---
        # Convert current adjacency matrix to a NumPy array
        adj = adjacency_dynamic_cpu[t].cpu().numpy()
        # Find indices of all edges (only upper triangle to avoid duplicates)
        i_indices, j_indices = np.nonzero(np.triu(adj, k=1))
        if len(i_indices) > 0:
            # Get the positions for the current time step (only x and y)
            pos_t = all_positions_cpu[t].cpu().numpy()  # shape: (node_amt, 3)
            # Build segments: each segment is defined by two endpoints ([x, y] pairs)
            segments = np.stack([pos_t[i_indices, :2], pos_t[j_indices, :2]], axis=1)  # shape: (n_edges, 2, 2)
            # Set the opacity (alpha) for each edge
            if ego_mask is not None:
                # For the current time step, recover the ego network indices
                ego_network_set = set(torch.nonzero(ego_mask[t], as_tuple=False).flatten().tolist())
                ego_list = list(ego_network_set)
                # Check which endpoints are in the ego network
                i_in = np.isin(i_indices, ego_list)
                j_in = np.isin(j_indices, ego_list)
                # If both endpoints are in the ego network, use higher opacity
                alphas = np.where(i_in & j_in, 0.25, 0.1)
            else:
                alphas = np.full(len(i_indices), 0.2)
            # Build an array of RGBA colors for each edge; base color is gray.
            base_color = np.array(mcolors.to_rgba("gray"))
            colors_arr = np.tile(base_color, (len(i_indices), 1))
            colors_arr[:, 3] = alphas  # Set the alpha channel for each edge
            # Create and add the LineCollection to the axes
            lc = LineCollection(segments, colors=colors_arr, linewidths=0.5)
            ax.add_collection(lc)

        # --- Plot nodes ---
        group_ids = all_positions_cpu[t, :, 2].long()
        unique_groups = torch.unique(group_ids)

        if ego_mask is not None:
            ego_network_set = set(torch.nonzero(ego_mask[t], as_tuple=False).flatten().tolist())
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
                        alpha=0.25,
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
            ax.scatter(ego_x, ego_y, s=200, marker='*',
                       c=colors[int(all_positions_cpu[t, ego_idx, 2].item()) % len(colors)],
                       zorder=10, alpha=0.5)
            ax.text(ego_x, ego_y, "Anchor", fontsize=12, color='black', zorder=11,
                    verticalalignment='bottom', horizontalalignment='right')
            
        if pred_groups is not None:
            i = 0
            for node in torch.nonzero(ego_mask.any(dim=0), as_tuple=False):
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
        ax.legend(loc="upper right", fontsize=8, frameon=False)


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=50, repeat=True)
    ani.save(save_path, writer=animation.PillowWriter(fps=10))
    print(f"Animation saved as {save_path}")

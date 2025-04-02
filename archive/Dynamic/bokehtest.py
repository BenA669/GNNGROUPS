import torch
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Slider, CustomJS, Label
from bokeh.layouts import column

def plot_faster_bokeh(all_positions_cpu, adjacency_dynamic_cpu, embed=None, 
                      ego_idx=None, ego_network_indices=None, pred_groups=None, 
                      ego_mask=None, save_path="bokeh_animation.html"):
    """
    Animate the dynamic graph using Bokeh. Instead of a GIF, an interactive HTML
    file is produced with a slider to navigate time steps. If an embedding is provided,
    spectral clustering is computed (unused for the animation). If ego info is provided,
    nodes/edges in the ego network are shown with higher opacity, and the ego node is highlighted.
    
    Parameters:
      all_positions_cpu: Tensor of shape (time_steps, node_amt, 3) with node positions and group id.
      adjacency_dynamic_cpu: Tensor of shape (time_steps, node_amt, node_amt) with adjacency info.
      embed (optional): An embedding tensor (currently used for spectral clustering).
      ego_idx (optional): Integer index for the ego node.
      ego_network_indices (optional): Either a list (or 1D tensor) of indices belonging to the ego network,
                                      or a 3D tensor (time_steps, n_ego, 3) from which indices will be recovered.
      pred_groups (optional): If provided, nodes in the ego network with an incorrect prediction are highlighted.
      ego_mask (optional): Unused here.
      save_path: Output HTML file path.
    """
    # Colors for groups:
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (Optional) Embedding part with spectral clustering.
    if embed is not None:
        from sklearn.cluster import SpectralClustering
        n_clusters = 4
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, 
                                                 affinity='nearest_neighbors', 
                                                 n_neighbors=2, random_state=42)
        predicted_labels = torch.from_numpy(
            spectral_clustering.fit_predict(embed)
        ).to(device=device)
        # (Further clustering-to-actual label matching is omitted here.)
    
    time_steps, node_amt, _ = all_positions_cpu.shape

    # Process ego network indices.
    if ego_network_indices is not None:
        if isinstance(ego_network_indices, torch.Tensor):
            if ego_network_indices.ndim == 3:
                candidate = ego_network_indices[0]  # shape: (n_ego, 3)
                full = all_positions_cpu[0]          # shape: (node_amt, 3)
                computed_indices = []
                for row in candidate:
                    match = (full == row).all(dim=1)
                    indices = torch.nonzero(match, as_tuple=False).flatten().tolist()
                    if indices:
                        computed_indices.append(indices[0])
                ego_network_indices = computed_indices
            else:
                ego_network_indices = ego_network_indices.tolist()
        elif isinstance(ego_network_indices, list):
            # Convert inner lists if needed.
            if len(ego_network_indices) > 0 and isinstance(ego_network_indices[0], list):
                ego_network_indices = [int(x[0]) for x in ego_network_indices]
        ego_network_set = set(ego_network_indices)
    else:
        ego_network_set = None

    # Precompute data for each frame.
    frames_nodes_ego = []       # For nodes in the ego network (or all if no ego network provided)
    frames_nodes_non_ego = []   # For nodes not in the ego network
    frames_edges = []           # Edge data for each time step
    frames_ego_node = []        # Ego node position (if provided)

    for t in range(time_steps):
        positions = all_positions_cpu[t]
        nodes_ego = {"x": [], "y": [], "color": []}
        nodes_non_ego = {"x": [], "y": [], "color": []}
        for i in range(node_amt):
            x = positions[i, 0].item()
            y = positions[i, 1].item()
            group = int(positions[i, 2].item())
            color = colors[group % len(colors)]
            if ego_network_set is not None:
                if i in ego_network_set:
                    nodes_ego["x"].append(x)
                    nodes_ego["y"].append(y)
                    nodes_ego["color"].append(color)
                else:
                    nodes_non_ego["x"].append(x)
                    nodes_non_ego["y"].append(y)
                    nodes_non_ego["color"].append(color)
            else:
                nodes_ego["x"].append(x)
                nodes_ego["y"].append(y)
                nodes_ego["color"].append(color)
        frames_nodes_ego.append(nodes_ego)
        frames_nodes_non_ego.append(nodes_non_ego)
        
        # Ego node data (if provided).
        ego_data = {"x": None, "y": None, "color": None}
        if ego_idx is not None and 0 <= ego_idx < node_amt:
            ego_data["x"] = positions[ego_idx, 0].item()
            ego_data["y"] = positions[ego_idx, 1].item()
            group = int(positions[ego_idx, 2].item())
            ego_data["color"] = colors[group % len(colors)]
        frames_ego_node.append(ego_data)
        
        # Compute edge data.
        edges = {"x0": [], "y0": [], "x1": [], "y1": [], "alpha": []}
        adj_t = adjacency_dynamic_cpu[t]
        for i in range(node_amt):
            for j in range(i+1, node_amt):
                if adj_t[i, j] == 1:
                    x0 = positions[i, 0].item()
                    y0 = positions[i, 1].item()
                    x1 = positions[j, 0].item()
                    y1 = positions[j, 1].item()
                    if ego_network_set is not None:
                        if (i in ego_network_set) and (j in ego_network_set):
                            edge_alpha = 0.25
                        else:
                            edge_alpha = 0.1
                    else:
                        edge_alpha = 0.2
                    edges["x0"].append(x0)
                    edges["y0"].append(y0)
                    edges["x1"].append(x1)
                    edges["y1"].append(y1)
                    edges["alpha"].append(edge_alpha)
        frames_edges.append(edges)

    # If pred_groups is provided, precompute misclassified nodes for ego network indices.
    if pred_groups is not None and ego_network_indices is not None:
        if torch.is_tensor(pred_groups):
            pred_groups = pred_groups.tolist()
        frames_misclassified = []
        # Use the group ids from time 0 as the "actual" labels.
        actual_groups = [int(all_positions_cpu[0, node, 2].item()) for node in ego_network_indices]
        for t in range(time_steps):
            positions = all_positions_cpu[t]
            mis = {"x": [], "y": []}
            for i, node in enumerate(ego_network_indices):
                if pred_groups[i] != actual_groups[i]:
                    mis["x"].append(positions[node, 0].item())
                    mis["y"].append(positions[node, 1].item())
            frames_misclassified.append(mis)
        mis_source = ColumnDataSource(data=frames_misclassified[0])
    else:
        frames_misclassified = None

    # Create initial ColumnDataSources.
    node_source_ego = ColumnDataSource(data=frames_nodes_ego[0])
    node_source_non_ego = ColumnDataSource(data=frames_nodes_non_ego[0])
    edge_source = ColumnDataSource(data=frames_edges[0])
    # For the ego node: if not provided, use empty lists.
    ego_node_data = (frames_ego_node[0] if frames_ego_node[0]["x"] is not None 
                     else {"x": [], "y": [], "color": []})
    ego_node_source = ColumnDataSource(data=ego_node_data)

    # Create the Bokeh figure.
    p = figure(title="Dynamic Graph Animation", x_axis_label="X", y_axis_label="Y",
               width=800, height=600)
    # Draw edges.
    p.segment(x0="x0", y0="y0", x1="x1", y1="y1", line_color="gray",
              line_alpha="alpha", source=edge_source)
    # Draw nodes not in ego network with lower opacity.
    p.circle(x="x", y="y", color="color", alpha=0.1, size=8,
             source=node_source_non_ego)
    # Draw ego network nodes with higher opacity.
    p.circle(x="x", y="y", color="color", alpha=0.7, size=8,
             source=node_source_ego)
    # Draw the ego node as a star.
    p.star(x="x", y="y", color="color", size=15, alpha=0.5,
           source=ego_node_source)
    
    # Add a label for the ego node (if provided).
    ego_label = None
    if ego_idx is not None and frames_ego_node[0]["x"] is not None:
        ego_label = Label(x=frames_ego_node[0]["x"], y=frames_ego_node[0]["y"],
                          text="Anchor", text_color="black", text_font_size="12pt")
        p.add_layout(ego_label)
    
    # If misclassified nodes exist, add them.
    if frames_misclassified is not None:
        p.circle(x="x", y="y", color="red", alpha=0.2, size=10,
                 source=mis_source)
    
    # Create a slider to move through time steps.
    slider = Slider(start=0, end=time_steps-1, value=0, step=1, title="Time Step")

    # CustomJS callback to update the plot when the slider value changes.
    callback_args = dict(
        slider=slider,
        node_source_ego=node_source_ego,
        node_source_non_ego=node_source_non_ego,
        edge_source=edge_source,
        ego_node_source=ego_node_source,
        frames_nodes_ego=frames_nodes_ego,
        frames_nodes_non_ego=frames_nodes_non_ego,
        frames_edges=frames_edges,
        frames_ego_node=frames_ego_node
    )
    # If misclassified nodes exist, add them to the callback args.
    if frames_misclassified is not None:
        callback_args["mis_source"] = mis_source
        callback_args["frames_misclassified"] = frames_misclassified
    # Also pass the ego label if available.
    if ego_label is not None:
        callback_args["ego_label"] = ego_label

    callback = CustomJS(args=callback_args, code="""
        var t = slider.value;
        node_source_ego.data = frames_nodes_ego[t];
        node_source_non_ego.data = frames_nodes_non_ego[t];
        edge_source.data = frames_edges[t];
        ego_node_source.data = (frames_ego_node[t].x != null) ? frames_ego_node[t] : {x: [], y: [], color: []};
        if (typeof frames_misclassified !== "undefined") {
            mis_source.data = frames_misclassified[t];
        }
        if (typeof ego_label !== "undefined") {
            ego_label.x = frames_ego_node[t].x;
            ego_label.y = frames_ego_node[t].y;
        }
        node_source_ego.change.emit();
        node_source_non_ego.change.emit();
        edge_source.change.emit();
        ego_node_source.change.emit();
        if (typeof mis_source !== "undefined") {
            mis_source.change.emit();
        }
    """)
    slider.js_on_change('value', callback)

    # Layout and output.
    layout = column(p, slider)
    output_file(save_path)
    save(layout)
    print(f"Bokeh animation saved as {save_path}")

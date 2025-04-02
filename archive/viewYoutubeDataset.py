import networkx as nx
import matplotlib.pyplot as plt
import random

# Load the dataset
def load_graph(edge_file):
    """Load the YouTube social network graph."""
    G = nx.read_edgelist(edge_file, nodetype=int)
    return G

def load_communities(community_file):
    """Load ground-truth communities."""
    communities = []
    with open(community_file, 'r') as f:
        for line in f:
            community = list(map(int, line.strip().split()))
            communities.append(community)
    return communities

# Assign colors to communities
def assign_community_colors(G, communities):
    """Assign a unique color to each community in the graph."""
    node_colors = {}
    for i, community in enumerate(communities):
        color = f"#{random.randint(0, 0xFFFFFF):06x}"  # Generate random hex color
        for node in community:
            node_colors[node] = color
    # Default color for nodes not in any community
    default_color = '#CCCCCC'
    return {node: node_colors.get(node, default_color) for node in G.nodes}

# Visualization
def visualize_graph(G, full_node_colors, sample_size=1000):
    """Visualize the graph with a subset of nodes and colored communities."""
    if sample_size and len(G) > sample_size:
        sampled_nodes = random.sample(list(G.nodes), sample_size)  # Convert nodes to a list for sampling
        G = G.subgraph(sampled_nodes)

    # Map the node colors to the current graph
    current_node_colors = [full_node_colors[node] for node in G.nodes]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Use a layout for better visualization
    nx.draw_networkx_nodes(G, pos, node_color=current_node_colors, node_size=20, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray", width=0.5)
    plt.title("YouTube Social Network with Ground-Truth Communities")
    plt.axis('off')
    plt.show()


# File paths (update these with the correct paths)
edge_file = "C:/Users/Benja/Documents/Projects/GNNGROUPS/GNNGROUPS/Data/youtube/com-youtube.ungraph.txt"  # Replace with the path to the edges file
community_file = "C:/Users/Benja/Documents/Projects/GNNGROUPS/GNNGROUPS/Data/youtube/com-youtube.top5000.cmty.txt"  # Replace with the path to the communities file

# Main
try:
    # Load graph and communities
    G = load_graph(edge_file)
    communities = load_communities(community_file)
    
    # Assign colors
    node_colors = assign_community_colors(G, communities)
    
    # Visualize
    visualize_graph(G, node_colors, sample_size=5000)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure the dataset files are downloaded and paths are correct.")
except KeyError as e:
    print(f"Error: Node {e} not found in color mapping.")
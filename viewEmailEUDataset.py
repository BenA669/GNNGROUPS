import networkx as nx
import matplotlib.pyplot as plt

def load_graph(edge_file):
    """
    Load the email-Eu-core network dataset as a directed graph.
    Each edge represents an email communication between nodes.
    """
    graph = nx.DiGraph()  # Directed graph
    with open(edge_file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue  # Skip comments
            source, target = map(int, line.strip().split()[:2])
            graph.add_edge(source, target)
    return graph

def load_labels(label_file):
    """
    Load department labels for nodes from the label file.
    Each line contains a node ID and its department ID.
    """
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue  # Skip comments
            node, department = map(int, line.strip().split())
            labels[node] = department
    return labels

def visualize_graph_with_groups(graph, labels, node_size=50, edge_alpha=0.3):
    """
    Visualize the graph with nodes colored by their department groups.
    """
    # Extract unique groups (departments)
    unique_groups = list(set(labels.values()))
    group_colors = {group: plt.cm.tab10(i % 10) for i, group in enumerate(unique_groups)}  # Color mapping

    # Assign colors to nodes based on their group
    node_colors = [group_colors[labels[node]] for node in graph.nodes]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, seed=42)  # Layout for consistent visualization
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, alpha=edge_alpha)

    # Add legend for the groups
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"Group {group}")
        for group, color in group_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Groups (Departments)", loc="best")

    plt.title("Email-Eu-core Network with Department Groups", fontsize=16)
    plt.axis("off")
    plt.show()

# File paths for the edge list and labels
edge_file = "Data/emailEU/email-Eu-core.txt"  # Update this path if the file is located elsewhere
label_file = "Data/emailEU/email-Eu-core-department-labels.txt"  # Update this path if the file is located elsewhere

# Load the graph and labels, and visualize
try:
    # Load graph
    email_graph = load_graph(edge_file)
    print(f"Loaded graph with {email_graph.number_of_nodes()} nodes and {email_graph.number_of_edges()} edges.")

    # Load labels
    node_labels = load_labels(label_file)
    print(f"Loaded labels for {len(node_labels)} nodes.")

    # Visualize graph with groups
    visualize_graph_with_groups(email_graph, node_labels)

except FileNotFoundError as e:
    print(f"File not found: {e}. Please check the file paths and try again.")

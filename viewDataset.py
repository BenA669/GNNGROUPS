import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as mcolors

def load_ego_facebook(ego_number):
    G = nx.Graph()
    path = f'Data/twitter/'
    
    # Load edges
    with open(path + f'{ego_number}.edges') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)

    # Load circles
    circles = []
    with open(path + f'{ego_number}.circles') as f:
        for line in f:
            circle = list(map(int, line.strip().split()[1:]))
            circles.append(circle)

    return G, circles

# Example usage
ego_id = 356963  # Replace with your desired ego network ID

# List to store unique ego node IDs
ego_nodes = set()

# Iterate through files in the dataset directory
for filename in os.listdir("Data/twitter/"):
    if filename.endswith('.edges'):
        # Extract the numeric prefix (ego node ID) and add to the set
        ego_id = filename.split('.')[0]
        ego_nodes.add(ego_id)

# Convert set to list
ego_nodes = list(ego_nodes)

# Check if there are any ego nodes found
if not ego_nodes:
    print("No ego nodes found in the specified directory.")
else:
    # Select a random ego node
    random_ego = random.choice(ego_nodes)
    print(f"Randomly selected ego node ID: {random_ego}")


G, circles = load_ego_facebook(random_ego)

# Generate a list of unique colors
colors = list(mcolors.TABLEAU_COLORS.values())
color_cycle = itertools.cycle(colors)

# Create a color map for nodes
node_color_map = {}

for circle in circles:
    color = next(color_cycle)
    for node in circle:
        node_color_map[node] = color

# Assign colors to nodes, defaulting to gray if not in any circle
node_colors = [node_color_map.get(node, 'gray') for node in G.nodes()]

# Draw the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # For consistent layout
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color='lightgray')
plt.title(f'Ego Network {ego_id} with Circles')
plt.show()

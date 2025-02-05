import torch
from model import TemporalGCN 
from makeEpisode import makeDatasetDynamic
from trainDyn import adjacency_to_edge_index
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from animate import plot_faster
from sklearn.cluster import SpectralClustering


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_dim = 3 
output_dim = 32  
num_nodes = 400 
num_timesteps = 20 
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


print("Model successfully loaded!")

positions, adjacency = makeDatasetDynamic()

time_steps, node_amt, _ = positions.shape

# Rearrange positions from [time_steps, node_amt, 3] to [node_amt, time_steps, 3]
positions_transposed = positions.permute(1, 0, 2)  # shape: [node_amt, time_steps, 3]
x = positions_transposed.unsqueeze(0).to(device)   # [1, node_amt, time_steps, 3]

# Convert adjacency to list of edge_indices
edge_indices = []
for t in range(time_steps):
    adj_t = adjacency[t]
    # Convert to edge_index
    edge_index_t = adjacency_to_edge_index(adj_t)
    edge_index_t = edge_index_t.to(device)
    edge_indices.append(edge_index_t)

# Forward pass
emb = model(x, edge_indices)  # shape: [1, node_amt, output_dim]



# predicted_labels = torch.from_numpy(hdb.fit_predict(output.detach().cpu().numpy())).to(device=device)
print(emb.shape)

# tsneEmbed = TSNE(perplexity=5).fit_transform(emb.cpu().detach().numpy()[0])


# fig = plt.figure(figsize=(10, 10)) 
# plt.title("Model Output Embeddings Visualized")
# plt.scatter(tsneEmbed[:, 0], tsneEmbed[:, 1], alpha=0.8)
# plt.show()

# plot(positions, adjacency)
plot_faster(positions, adjacency, emb)
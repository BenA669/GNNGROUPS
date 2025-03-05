import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn

# Define GCNConv model
class GCNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNModel, self).__init__()
        self.gcn = GCNConv(input_dim, output_dim) # Define GCN Layer

    def forward(self, features_input, adjacency_input):
        output = self.gcn(features_input, adjacency_input)
        return output

# Make random data based on provided number of nodes
def makeData(node_amount, feature_dimension):
    # Randomize number of edges
    num_edges_rand = torch.randint(10, 20, (1,))

    # Feature Matrix of shape (Node Amount, Feature Dimension)
    features = torch.rand(node_amount, feature_dimension)

    # Edge Index of shape (2, Number of edges)
    edge_index = torch.randint(0, node_amount, (2, num_edges_rand))
    return features, edge_index
    


input_dim = 2
output_dim = 8
model = GCNModel(input_dim, output_dim)





# Dataset for node amount 10:
data_node_10 = makeData(node_amount=10, feature_dimension=input_dim)

# Dataset for node amount 20:
data_node_20 = makeData(node_amount=20, feature_dimension=input_dim)

# Dataset for node amount 30:
data_node_30 = makeData(node_amount=30, feature_dimension=input_dim)

print("Feature Matrix 10 Nodes Shape: {}".format(data_node_10[0].shape))
print("Edge Index 10 Nodes Shape: {}\n".format(data_node_10[1].shape))

print("Feature Matrix 20 Node Shape: {}".format(data_node_20[0].shape))
print("Edge Index 20 Node Shape: {}\n".format(data_node_20[1].shape))

print("Feature Matrix 30 Node Shape: {}".format(data_node_30[0].shape))
print("Edge Index 30 Node Shape: {}\n".format(data_node_30[1].shape))

# Model passes
model_output = model(data_node_10[0], data_node_10[1])
print("\nModel output shape 10 Nodes: {}".format(model_output.shape))

model_output = model(data_node_20[0], data_node_20[1])
print("Model output shape 20 Nodes: {}".format(model_output.shape))

model_output = model(data_node_30[0], data_node_30[1])
print("Model output shape 30 Nodes: {}".format(model_output.shape))
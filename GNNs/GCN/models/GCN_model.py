import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader


class MatrixDataset(Dataset):
    def __init__(self, list_matrices):
        self.list_matrices = list_matrices

    def __len__(self):
        return len(self.list_matrices)

    def __getitem__(self, idx):
        item = self.list_matrices[idx]

        # Assume mstype is always the second-to-last item
        mstype = item[1]
        # All items before mstype and edds are matrices
        matrices = item[0]

        # Convert mstype to indices
        types = {-1: 0, 0: 1, 1: 2, 2: 3}
        mstype = types[mstype]

        
        return matrices, mstype


class Simple_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Simple_GCN, self).__init__()
        # First graph convolutional layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Second graph convolutional layer
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph connectivity in COO format [2, num_edges]
        
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Optional: Add dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer (output layer)
        x = self.conv2(x, edge_index)
        
        return x
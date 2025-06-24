import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GraphNorm


class GCN_dropout(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, node_dropout=0, edge_dropout=0):
        super(GCN_dropout, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)# Add more layers and complexity
        self.conv3 = GCNConv(hidden_channels, hidden_channels*2)
        self.lin1 = torch.nn.Linear(114*hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels*2)
        self.norm1 = GraphNorm(hidden_channels)
        self.norm2 = GraphNorm(hidden_channels//2)
        self.norm3 = GraphNorm(hidden_channels*2)

        # Dropout parameters
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
    def feature_node_dropout(self, x, dropout_rate):
        """Apply feature masking to nodes (keeps same number of nodes)"""
        if not self.training or dropout_rate == 0:
            return x
            
        # Create node mask
        node_mask = torch.rand(x.size(0), device=x.device) > dropout_rate
        node_mask = node_mask.float().unsqueeze(1)  # Shape: [num_nodes, 1]
        
        # Apply mask to features (broadcast across feature dimensions)
        x_masked = x * node_mask
        
        return x_masked
    
    def feature_edge_dropout(self, x, edge_index, dropout_rate):
        """Apply edge masking during message passing (keeps same number of edges)"""
        if not self.training or dropout_rate == 0:
            return edge_index, None
            
        # Create edge mask
        num_edges = edge_index.size(1)
        edge_mask = torch.rand(num_edges, device=edge_index.device) > dropout_rate
        
        return edge_index, edge_mask
    
    def masked_gat_conv(self, conv_layer, x, edge_index, edge_mask=None):
        """GAT convolution with optional edge masking"""
        if edge_mask is not None:
            # Create a modified edge_index with masked edges set to self-loops
            masked_edge_index = edge_index.clone()
            
            # For masked edges, replace with self-loops to maintain structure
            num_nodes = x.size(0)
            masked_edges = ~edge_mask
            
            if masked_edges.sum() > 0:
                # Replace masked edges with self-loops (or duplicate existing edges)
                masked_edge_index[0, masked_edges] = masked_edge_index[0, masked_edges]  # Self-loop
                masked_edge_index[1, masked_edges] = masked_edge_index[0, masked_edges]  # Self-loop
            
            return conv_layer(x, masked_edge_index)
        else:
            return conv_layer(x, edge_index)
        
    def forward(self, x, edge_index, batch=None):
        # Apply feature-based dropout (maintains structure)
        if self.training:
            # Apply node feature dropout
            x = self.feature_node_dropout(x, self.node_dropout)
            
            # Get edge mask for edge dropout
            edge_index, edge_mask = self.feature_edge_dropout(x, edge_index, self.edge_dropout)
        else:
            edge_mask = None
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = self.batch_norm1(x)
        x = self.norm1(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.norm2(x, batch)
        #x = self.batch_norm2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        '''# Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.norm3(x)
        x = F.dropout(x, p=0.5, training=self.training)'''
        
        
        x = x.reshape(-1, 228*x.shape[1])
        #x = global_mean_pool(x, batch)  # Global max pooling
        #x_max = global_max_pool(x, batch)  # Global max pooling
        #x = torch.cat([x_mean, x_max], dim=1)  # Concatenate mean and max pooled features
        
        # MLP head
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.batch_norm1(x)
        x = self.lin2(x)
        
        return x
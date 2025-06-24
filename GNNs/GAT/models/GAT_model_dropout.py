import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader




class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=6, node_dropout=0.1, edge_dropout=0.1):
        super(GAT, self).__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels , heads=heads, concat=True)
        self.conv4 = GATConv(hidden_channels*heads, hidden_channels , heads=heads, concat=True)
        self.conv5 = GATConv(hidden_channels*heads, hidden_channels , heads=heads, concat=True)
        self.lin1 = torch.nn.Linear(228*hidden_channels*heads, hidden_channels )
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
        self.batch_norm11 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.batch_norm12 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.batch_norm13 = torch.nn.BatchNorm1d(hidden_channels * heads)

        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels // 4)
        self.norm1 = GraphNorm(hidden_channels * heads)
        self.norm2 = GraphNorm(hidden_channels*heads)
        self.norm3 = GraphNorm(hidden_channels*heads)
        self.norm4 = GraphNorm(hidden_channels*heads)
        self.norm5 = GraphNorm(hidden_channels*heads)
        

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
        
        # First GAT layer
        x = self.conv1(x, edge_index)  
        x = F.elu(x)
        #x = F.relu(x)
        x = self.norm1(x, batch)
        #x = self.batch_norm11(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        #x = F.relu(x)
        x = self.norm2(x, batch)
        #x = self.batch_norm12(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Third GAT layer
        '''
        x = self.conv3(x, edge_index)

        x = F.relu(x)
        x = self.norm3(x, batch)
        #x = self.batch_norm13(x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.norm4(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.norm5(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)'''

        
        
        
        
        '''x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)'''
        #x = global_mean_pool(x, batch)
        # MLP head
        x = x.reshape(-1,228*self.hidden_channels*self.heads)#
        
        embedding = x

        x = self.lin1(x)
        x = F.elu(x)
        x = self.batch_norm1(x)
        
        #x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.lin2(x)
        
        return x
    
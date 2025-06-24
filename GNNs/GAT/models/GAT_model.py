import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels , heads=heads, concat=True)
        self.conv4 = GATConv(hidden_channels*heads, hidden_channels , heads=heads, concat=True)
        self.conv5 = GATConv(hidden_channels*heads, hidden_channels , heads=heads, concat=True)
        self.lin1 = torch.nn.Linear(152*hidden_channels*heads, hidden_channels )
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
        
    def forward(self, x, edge_index, batch=None):
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
        
        '''x = self.conv3(x, edge_index)

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
        #x = global_max_pool(x, batch)
        # MLP head
        x = x.reshape(-1, 152*x.shape[1])#
        embedding = x
        
        return x, embedding
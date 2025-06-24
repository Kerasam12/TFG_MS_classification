import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, Counter
import random

import sys
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Data_Preprocessing")
from data_cleaning import create_clean_dataset


from CreateMultilayerMS import create_multilayer_ms

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from graph_prunning import graph_prunning_threshold, normalize_and_threshold
from DataAugmentation import datalist_mixing,datalist_mixing_balanced






# BARCELONA DATA 
route_patients = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES"
route_patients_FA_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FA/"
route_patients_Func_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FUNC/"
route_patients_GM_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/GM_networks/"
route_classes_bcn = "Dades/DADES_BCN/DADES_HCB/subject_clinical_data_20201001.xlsx"

#NAPLES DATA 
route_patients_FA_nap = "Dades/DADES_NAP/Naples/DTI_networks/"
route_patients_Func_nap = "Dades/DADES_NAP/Naples/rsfmri_networks/"
route_patients_GM_nap = "Dades/DADES_NAP/Naples/GM_networks/"
route_classes_nap = "Dades/DADES_NAP/Naples/naples2barcelona_multilayer.xlsx"


def generate_train_test_datalist(data_list, train_ratio = 0.7, random_state = 42):
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Group data by class (mstype)
    class_groups = defaultdict(list)
    for item in data_list:
        mstype = item[3]  # The class label is at index 3

        class_groups[mstype].append(item)
    
    # Initialize empty lists for train and test sets
    train_list = []
    test_list = []
    
    # For each class, split according to the train_ratio
    for mstype, items in class_groups.items():
        # Shuffle the items for this class
        items_shuffled = items.copy()
        random.shuffle(items_shuffled)
        
        # Calculate split point
        split_idx = int(len(items_shuffled) * train_ratio)
        
        # Split items for this class
        train_items = items_shuffled[:split_idx]
        test_items = items_shuffled[split_idx:]
        
        # Add to final lists
        train_list.extend(train_items)
        test_list.extend(test_items)
    
    # Shuffle the final lists to mix different classes
    random.shuffle(train_list)
    random.shuffle(test_list)
    
    # Print class distribution for verification
    print("Original dataset class distribution:")
    original_counts = Counter([item[3] for item in data_list])
    total_original = len(data_list)
    for cls, count in sorted(original_counts.items()):
        percentage = count / total_original * 100
        print(f"  Class {cls}: {count} samples ({percentage:.2f}%)")
    
    print("\nTrain set class distribution:")
    train_counts = Counter([item[3] for item in train_list])
    total_train = len(train_list)
    for cls, count in sorted(train_counts.items()):
        percentage = count / total_train * 100
        print(f"  Class {cls}: {count} samples ({percentage:.2f}%)")
    
    print("\nTest set class distribution:")
    test_counts = Counter([item[3] for item in test_list])
    total_test = len(test_list)
    for cls, count in sorted(test_counts.items()):
        percentage = count / total_test * 100
        print(f"  Class {cls}: {count} samples ({percentage:.2f}%)")
    
    # Verify the split ratio is maintained for each class
    print("\nVerifying class-wise split ratios:")
    for cls in original_counts.keys():
        orig_count = original_counts[cls]
        train_count = train_counts.get(cls, 0)
        train_ratio_actual = train_count / orig_count if orig_count > 0 else 0
        print(f"  Class {cls}: {train_ratio_actual:.2f} (target: {train_ratio:.2f})")
    
    print(f"\nFinal train set size: {len(train_list)} ({len(train_list)/len(data_list)*100:.2f}%)")
    print(f"Final test set size: {len(test_list)} ({len(test_list)/len(data_list)*100:.2f}%)")
    
    return train_list, test_list

class GraphDataset(Dataset):
    """
    Dataset class for graph classification task.
    
    Each datapoint contains:
    - adjacency matrix (76x76)
    - graph type (-1, 0, 1, 2) which is converted to (0, 1, 2, 3)
    """
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the adjacency matrix and label from the data
        adj_matrix = self.data[idx][0]  # 76x76 adjacency matrix
        graph_type = self.data[idx][1]  # Original label (-1, 0, 1, 2)
        
        # Convert graph type to appropriate index (0, 1, 2, 3)
        if graph_type == -1:
            graph_type = 0
        elif graph_type == 0:
            graph_type = 1
        elif graph_type == 1:
            graph_type = 2
        elif graph_type == 2:
            graph_type = 3
        
        # Convert to PyTorch tensors
        adj_matrix = torch.FloatTensor(adj_matrix)
        graph_type = torch.LongTensor([graph_type])[0]  # Convert to scalar tensor
        
        return adj_matrix, graph_type


class GraphKernelLayer(nn.Module):
    """
    Graph Kernel Layer: Performs graph convolution operation.
    """
    def __init__(self, in_features, out_features):
        super(GraphKernelLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        """
        x: Node features (batch_size, num_nodes, in_features)
        adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # Normalize adjacency matrix
        adj_normalized = self.normalize_adj(adj)

        

        # Graph convolution: adj_normalized @ x @ weight
        support = torch.matmul(x, self.weight)  # (batch_size, num_nodes, out_features)
        output = torch.matmul(adj_normalized, support)  # (batch_size, num_nodes, out_features)
        
        return output
    
    def normalize_adj(self, adj):
        """
        Symmetrically normalize adjacency matrix.
        """
        # Add self-loops
        batch_size, n, _ = adj.size()
        identity = torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1).to(adj.device)
        adj = adj + identity
        
        # Compute D^(-1/2) * A * D^(-1/2)
        rowsum = torch.sum(adj, dim=2)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


class GraphKernelNetwork(nn.Module):
    """
    Graph Kernel Network for graph classification.
    """
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4, num_layers=5, dropout=0.5):
        super(GraphKernelNetwork, self).__init__()
        
        # Initial node feature embedding
        # Since we only have adjacency matrix, we'll create simple node features
        self.node_embedding = nn.Parameter(torch.FloatTensor(152, input_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphKernelLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GraphKernelLayer(hidden_dim, hidden_dim))
        
        # Readout (graph-level pooling)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, adj):
        """
        adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size = adj.size(0)
        
        # Initialize node features (same initial features for all graphs in batch)
        x = self.node_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = F.relu(conv(x, adj))
        
        # Global pooling (mean of node features)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)
        
        # MLP for classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_dataloader(data_list, batch_size=32, shuffle=True):
    """
    Create a dataloader from a list of graph data.
    
    Args:
        data_list: List of data points where each item contains [adj_matrix, graph_type, ...]
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader object
    """
    dataset = GraphDataset(data_list)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    return dataloader


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create dataset
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    results = []
    

    print("Total number of dataopoints:",len(datalist))
    ms_types = [datapoint[3] for datapoint in datalist]
        
    print(f"NO MS: {ms_types.count(-1)}, Percentage: {ms_types.count(-1) / len(ms_types):.2f}")
    print(f"RRMS: {ms_types.count(0)}, Percentage: {ms_types.count(0) / len(ms_types):.2f}")
    print(f"SPMS: {ms_types.count(1)}, Percentage: {ms_types.count(1) / len(ms_types):.2f}")
    print(f"PPMS: {ms_types.count(2)}, Percentage: {ms_types.count(2) / len(ms_types):.2f}")
    print(f"MS types: {np.unique(ms_types)} \n")

    datalist_train, datalist_test = generate_train_test_datalist(datalist, train_ratio = 0.7)
    
    # Prunning parameters
    prunning_threshold = 0.7
    mixing_levels = [0, 1, 2, 3]

    # Normalize and trheshold
    datalist_train = normalize_and_threshold(datalist_train, prunning_threshold)
    datalist_test = normalize_and_threshold(datalist_test, prunning_threshold)

    mixed_datalist = datalist_mixing_balanced(datalist_train, mixing_levels,200)

    datalist_new = create_multilayer_ms(mixed_datalist)
    datalist_test1 = create_multilayer_ms(datalist_test)
    

    print("Total number of dataopoints:",len(datalist_new))
    print("total number of dataopoints:",len(datalist_test1))
    # Create dataloader
    train_loader = create_dataloader(datalist_new, batch_size=8)
    test_loader = create_dataloader(datalist_test1, batch_size=16, shuffle=False)
    
    # Initialize model
    model = GraphKernelNetwork()
    model = model.to(device)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.75, 9, 20, 4]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=5e-3)
    
    # Training loop (example)
    def train(model, loader, optimizer, criterion, epochs=10):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for adj, labels in loader:
                adj = adj.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = model(adj)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
            
    # Training loop (example)
    def test(model, loader, criterion):
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for adj, labels in loader:
            adj = adj.to(device)
            labels = labels.to(device)
            output = model(adj)
            loss = criterion(output, labels)

            
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
        
        print(f' Loss: {total_loss/len(loader):.4f}, '
                f'Accuracy: {100 * correct / total:.2f}%'
                f' Confusion Matrix: {confusion_matrix(labels.cpu(), predicted.cpu())}')
     

    
    # Train the model
    train(model, train_loader, optimizer, criterion, epochs=50)
    test(model, test_loader, criterion)
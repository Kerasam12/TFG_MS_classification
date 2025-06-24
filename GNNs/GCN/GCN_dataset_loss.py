import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GraphNorm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

import os
import csv

import sys
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Data_Preprocessing")

from data_cleaning import create_clean_dataset
from graph_prunning import normalize_and_threshold
from DataAugmentation import datalist_mixing, datalist_mixing_balanced
from CreateMultilayerMS import create_multilayer_ms
from GraphMetrics import open_features_dict, create_graph_list
from GraphMetricsIndividual import create_graph_list_individual

# BARCELONA DATA 
route_patients = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES"
route_patients_FA_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FA/"
route_patients_Func_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FUNC/"
route_patients_GM_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/GM_networks/"

route_controls_FA_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/CONTROLES/FA/"
route_controls_Func_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/CONTROLES/FUNC/"
route_controls_GM_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/CONTROLES/GM_networks/"
route_classes_bcn = "Dades/DADES_BCN/DADES_HCB/subject_clinical_data_20201001.xlsx"

route_features_controls_bcn = "Dades/DADES_BCN/DADES_HCB/VOLUM_nNODES_CONTROLS.xls"
route_features_patients_bcn = "Dades/DADES_BCN/DADES_HCB/VOLUM_nNODES_PATIENTS.xls"

#NAPLES DATA 
route_patients_FA_nap = "Dades/DADES_NAP/Naples/DTI_networks/"
route_patients_Func_nap = "Dades/DADES_NAP/Naples/rsfmri_networks/"
route_patients_GM_nap = "Dades/DADES_NAP/Naples/GM_networks/"
route_classes_nap = "Dades/DADES_NAP/Naples/naples2barcelona_multilayer.xlsx"

route_features_train = "/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/computed_graphs_train/"
route_features_test = "/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/computed_graphs_test/"   


def separate_train_test(datalist, file_path_train, file_path_test):
    """
    Separate the datalist into training and testing sets.
    
    Parameters:
    datalist: List of tuples (graph, mstype, edds, name)
    file_path_train: Path to the training data file
    file_path_test: Path to the testing data file
    
    Returns:
    train_data: Training data
    test_data: Testing data
    """
    train_data = []
    test_data = []
    
    file_names_train = os.listdir(file_path_train)
    file_names_test = os.listdir(file_path_test)
    
    file_names_train = [name[:-8] for name in file_names_train if name.endswith('.gpickle')]
    file_names_test = [name[:-8] for name in file_names_test if name.endswith('.gpickle')]
    for matrix1,matrix2, matrix3, mstype, edds, name, dataset_name in datalist:
        # Check if the name is in the training or testing file list
        name_r = 'r' + name  
        # Remove the .gpickle extension from the name
        if name in file_names_train:
            train_data.append((matrix1,matrix2,matrix3, mstype, edds, name,dataset_name))
        elif name in file_names_test:
            test_data.append((matrix1,matrix2,matrix3,mstype, edds, name, dataset_name))

        elif name_r in file_names_train:
            train_data.append((matrix1,matrix2,matrix3, mstype, edds, name,dataset_name))
        elif name_r in file_names_test:
            test_data.append((matrix1,matrix2,matrix3, mstype, edds, name,dataset_name))
        else:
            print(f"Name {name}, {name_r} not found in either training or testing files.")

            
    
    return train_data, test_data

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, num_datasets = 2, node_dropout=0.3, edge_dropout=0.4):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)# Add more layers and complexity
        self.conv3 = GCNConv(hidden_channels, hidden_channels*2)
        self.lin1 = torch.nn.Linear(114*hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
        self.lin3 = torch.nn.Linear(hidden_channels, num_datasets)
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
        x_dataset = x.clone()  # Create a copy for dataset prediction
        #x = global_mean_pool(x, batch)  # Global max pooling
        #x_max = global_max_pool(x, batch)  # Global max pooling
        #x = torch.cat([x_mean, x_max], dim=1)  # Concatenate mean and max pooled features
        
        # MLP head to predict classes 
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.batch_norm1(x)
        x = self.lin2(x)

        #MLP head to predict dataset
        x_dataset = self.lin1(x_dataset)
        x_dataset = F.elu(x_dataset)
        x_dataset = F.dropout(x_dataset, p=0.5, training=self.training)
        x_dataset = self.batch_norm1(x_dataset)
        x_dataset = self.lin3(x_dataset)
        
        return x, x_dataset

def create_gcn_dataset(graphlist, features_dict):
    """
    Create a PyTorch Geometric dataset from a list of NetworkX graphs.
    
    Parameters:
    graphlist: List of tuples (graph, mstype, edds, name)
    
    Returns:
    dataset: List of PyTorch Geometric Data objects
    """
    dataset = []
    
    for graph, mstype, edds, name,dataset_name in graphlist:

        convert_types = {-1:3, 0:0, 1:1, 2:2}
        convert_names = {"BCN":0, "NAP":1}
        # Extract node features as a matrix
        # We'll use degree, strength, betweenness, clustering and volumes as node features
        node_features = features_dict[name]
        
        # Standardize each feature column independently using z-score
        for col in range(node_features.shape[1]):  # Iterate through all the columns
            col_mean = np.mean(node_features[:, col])
            col_std = np.std(node_features[:, col])
            if col_std == 0:
                # If standard deviation is zero, set to zero to avoid division by zero
                node_features[:, col] = 0
                print(f"Column {col} has zero standard deviation, setting to zero.")
            else:
                # Standardize the column
                node_features[:, col] = (node_features[:, col] - col_mean) / col_std
        
        
          
        # Convert to PyTorch tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edges and create edge_index
        edge_list = list(graph.edges())
        if not edge_list:
            # If no edges, create a self-loop for each node
            edge_list = [(i, i) for i in range(len(node_features))]

        edge_list = [(int(u), int(v)) for u, v in edge_list]   
        edges_transposed = list(zip(*edge_list))  # Convert to [2, num_edges] format
        edge_index = torch.tensor(edges_transposed, dtype=torch.long)
        
        # Extract edge weights if available
        edge_attr = None
        
        edge_weights = [float(graph[str(u)][str(v)].get('weight', 0)) for u, v in edge_list]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        # Extract label (mstype)
        # Convert labels to integers
        names_dataset = convert_names[dataset_name]
        
        mstype = convert_types[mstype]
        y = torch.tensor(mstype, dtype=torch.long).view(1)
        y_dataset = torch.tensor(names_dataset, dtype=torch.long).view(1)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_dataset = y_dataset)
        data.name = name  # Store name as attribute
        
        dataset.append(data)
    
    return dataset

def create_dataloader(dataset, batch_size=16, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train, validation, and test dataloaders from a dataset.
    
    Parameters:
    dataset: List of PyTorch Geometric Data objects
    batch_size: Batch size for dataloaders
    test_size: Fraction of data to use for testing
    val_size: Fraction of training data to use for validation
    random_state: Random seed for reproducibility
    
    Returns:
    train_loader, val_loader, test_loader: PyTorch Geometric DataLoader objects
    """
    # Split dataset into train+val and test
    train_val_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=random_state, 
        stratify=[data.y.item() for data in dataset]
    )
    
    # Split train+val into train and val
    train_dataset, val_dataset = train_test_split(
        train_val_dataset, test_size=val_size/(1-test_size), random_state=random_state,
        stratify=[data.y.item() for data in train_val_dataset]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def create_train_val_dataloader(dataset, batch_size=16, val_size=0.2, random_state=42):
    """
    Create train, validation, and test dataloaders from a dataset.
    
    Parameters:
    dataset: List of PyTorch Geometric Data objects
    batch_size: Batch size for dataloaders
    test_size: Fraction of data to use for testing
    val_size: Fraction of training data to use for validation
    random_state: Random seed for reproducibility
    
    Returns:
    train_loader, val_loader, test_loader: PyTorch Geometric DataLoader objects
    """
    # Split dataset into train+val and test
    
    # Split train+val into train and val
    '''train_dataset, val_dataset = train_test_split(
        dataset, test_size=val_size, random_state=random_state,
        stratify=[data.y.item() for data in dataset]
    )'''
    
    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size)
   
    
    print(f"Training samples: {len(dataset)}")
    #print(f"Validation samples: {len(val_dataset)}")
    
    
    return train_loader


def create_test_dataloader(dataset, batch_size=16, val_size=0.3, random_state=42):
    test_dataset, val_dataset = train_test_split(
        dataset, test_size=val_size, random_state=random_state,
        stratify=[data.y.item() for data in dataset]
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f"Test samples: {len(dataset)}")
    return test_loader, val_loader

def train_gcn_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """
    Train a GCN model on the given data.
    
    Parameters:
    model: GCN model
    train_loader: Training data loader
    val_loader: Validation data loader
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimization
    
    Returns:
    model: Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.75, 9, 20, 4], dtype=torch.float).to(device))  # Adjust weights as needed weight=torch.tensor([1.75, 9, 20, 4], dtype=torch.float).to(device)
    criterion_dataset = torch.nn.CrossEntropyLoss()  # Adjust weights for dataset prediction
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3000, T_mult=2, eta_min=0.000001)
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_loss_out = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, pred_dataset = model(data.x, data.edge_index, data.batch)
            loss_out = criterion(out, data.y)
            loss_dataset =  criterion_dataset(pred_dataset, data.y_dataset)
            loss = loss_out - 0.5*loss_dataset  # Combine losses
            
            loss.backward()  # Backpropagate the loss for the output
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_out += loss_out.item()
            _, predicted = out.max(1)
            total += data.y.size(0)
            correct += predicted.eq(data.y).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_loss_out = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, pred_dataset = model(data.x, data.edge_index, data.batch)
                loss_out = criterion(out, data.y)
                loss_dataset = criterion_dataset(pred_dataset, data.y_dataset)
                loss = loss_out - 0.5*loss_dataset  # Combine losses
                val_loss += loss.item()
                val_loss_out += loss_out.item()
                _, predicted = out.max(1)
                total += data.y.size(0)
                correct += predicted.eq(data.y).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss_out/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss_out/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save the best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gcn_model.pth')
    
    # Load the best model
    #model.load_state_dict(torch.load('best_gcn_model.pth'))
    return model

def test_gcn_model(model, test_loader):
    """
    Test a trained GCN model.
    
    Parameters:
    model: Trained GCN model
    test_loader: Test data loader
    
    Returns:
    test_acc: Test accuracy
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_y = []
    all_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, pred_dataset = model(data.x, data.edge_index, data.batch)
            _, predicted = out.max(1)

            # Move predictions and targets to CPU for sklearn metrics
            all_y.extend(data.y.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_y, all_pred)
    
    # For multi-class classification, we need to specify averaging method
    precision = precision_score(all_y, all_pred, average='weighted', zero_division=0)
    recall = recall_score(all_y, all_pred, average='weighted', zero_division=0)
    f1 = f1_score(all_y, all_pred, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_y, all_pred)
    
    # Print results
    print(f'Test Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:')
    print(conf_matrix)
    # Calculate metrics per class
    class_metrics = {}
    for class_label in sorted(set(all_y)):
        class_precision = precision_score(all_y, all_pred, labels=[class_label], average=None, zero_division=0)[0]
        class_recall = recall_score(all_y, all_pred, labels=[class_label], average=None, zero_division=0)[0]
        class_f1 = f1_score(all_y, all_pred, labels=[class_label], average=None, zero_division=0)[0]
        class_metrics[class_label] = {
            "Precision": class_precision,
            "Recall": class_recall,
            "F1 Score": class_f1
        }
        print(f"Class {class_label}: Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1 Score: {class_f1:.4f}")

    # Save metrics to a CSV file
    with open("GCN_data_augment_class_metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "F1 Score"])
        for class_label, metrics in class_metrics.items():
            writer.writerow([class_label, metrics["Precision"], metrics["Recall"], metrics["F1 Score"]])
    print("Class metrics saved to class_metrics.csv")

    return accuracy

if __name__ == "__main__":
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, 
                                    route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, 
                                    route_classes_bcn, route_classes_nap)
    datalist_thresholded = normalize_and_threshold(datalist, 0.5)
    datalist_train,datalist_test = separate_train_test(datalist_thresholded, route_features_train, route_features_test)
    print(f"Training samples: {len(datalist_train)}")
    datalist_augmented = datalist_mixing_balanced(datalist_train, [0,1,2,3], 110)

    # 1. Generate graphs from datalist
    graphlist_train = create_graph_list_individual(datalist_train)
    graphlist_test = create_graph_list_individual(datalist_test)
    

    # 2. Load features from disk
    features_dict_train = open_features_dict("computed_features_train_individual", train = True)
    features_dict_test = open_features_dict("computed_features_test_individual", train = False)
    
    print(f"Training samples: {len(graphlist_train)}")
    print(f"Testing samples: {len(graphlist_test)}")
    # Create dataset and dataloaders
    dataset_train = create_gcn_dataset(graphlist_train, features_dict_train)
    print(f"Training samples: {len(dataset_train)}")

    dataset_test = create_gcn_dataset(graphlist_test, features_dict_test)
    print(f"Testing samples: {len(dataset_test)}")
    train_loader = create_train_val_dataloader(dataset_train, batch_size=96)
    test_loader, val_loader = create_test_dataloader(dataset_test, batch_size=96)


    # Initialize and train the model
    num_node_features = dataset_train[0].x.shape[1]  # Number of node features
    

    hidden_channels = 128
    # Calculate number of classes by finding unique labels
    unique_labels = set()
    for _, mstype, _, _,_ in graphlist_train:
        unique_labels.add(mstype)
    num_classes = len(unique_labels)
    
    model = GCN(num_node_features, hidden_channels, num_classes)
    trained_model = train_gcn_model(model, train_loader, val_loader, num_epochs=3000, learning_rate = 0.00001)
    

    best_model_path = 'best_gcn_model.pth'
    best_trained_model = GCN(num_node_features, hidden_channels, num_classes)
    best_trained_model.load_state_dict(torch.load(best_model_path))
    best_trained_model = best_trained_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Test the model
    print("\nTest trained model")
    test_acc = test_gcn_model(trained_model, test_loader)
    print("\nTest best Model")
    test_acc_best = test_gcn_model(best_trained_model, test_loader)

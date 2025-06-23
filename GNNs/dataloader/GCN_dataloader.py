import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GraphNorm


from sklearn.model_selection import train_test_split



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



def create_gcn_dataset(graphlist, features_dict, convert_types = {-1:3, 0:0, 1:1, 2:2}):
    """
    Create a PyTorch Geometric dataset from a list of NetworkX graphs.
    
    Parameters:
    graphlist: List of tuples (graph, mstype, edds, name)
    
    Returns:
    dataset: List of PyTorch Geometric Data objects
    """
    dataset = []
    
    for graph, mstype, edds, name,dataset_name in graphlist:

        # Extract node features as a matrix
        # We'll use degree, strength, betweenness, clustering and volumes as node features
        node_features = features_dict[name]
        
        # Standardize each feature column independently using z-score
        for col in range(node_features.shape[0]):  # Iterate through all the columns
            col_mean = np.mean(node_features[col, :])
            col_std = np.std(node_features[col, :])
            if col_std == 0:
                # If standard deviation is zero, set to zero to avoid division by zero
                print("column_values", node_features[col, :])
                node_features[col, :] = 0
                print(f"Column {col} has zero standard deviation, setting to zero.")
                
                
            else:
                # Standardize the column
                node_features[col, :] = (node_features[col, :] - col_mean) / col_std
        
        

        
          
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
        mstype = convert_types[mstype]
        y = torch.tensor(mstype, dtype=torch.long).view(1)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
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

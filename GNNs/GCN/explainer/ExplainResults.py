import torch
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
import numpy as np
CUDA_LAUNCH_BLOCKING=1 


import torch.nn.functional as F
import matplotlib.pyplot as plt

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
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/")

from Data_Preprocessing.data_cleaning import create_clean_dataset
from Data_Preprocessing.graph_prunning import normalize_and_threshold
from Data_Preprocessing.DataAugmentation import datalist_mixing, datalist_mixing_balanced
from Data_Preprocessing.CreateMultilayerMS import create_multilayer_ms
from Data_Preprocessing.GraphMetrics import open_features_dict, create_graph_list
from Data_Preprocessing.GraphMetricsIndividual import create_graph_list_individual

from GNNs.dataloader.GCN_dataloader import create_gcn_dataset, create_train_val_dataloader, create_test_dataloader, separate_train_test
from GNNs.train_test_pipeline.train_test_pipeline import train_gcn_model, test_gcn_model

from GNNs.GCN.models.GCN_model_dropout import GCN_dropout

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
    

def explain_model(model, test_loader, heatmap_name ='feature_importance_heatmap.png'):
    dataset = test_loader.dataset  # Assuming test_loader is defined
    data = dataset[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Initialize GNN explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    # Explain specific nodes
    node_indices = range(data.x.shape[0])  # Choose nodes to explain
    print(f"Number of nodes to explain: {node_indices}")
    print("Generating explanations...")

    # Generate explanation for the node
    explanation = explainer(data.x, data.edge_index)

    for node_idx in node_indices:
        #print(f"\nExplaining node {node_idx}:")
    
        # Feature importance (node_mask)
        feature_importance = explanation.node_mask[node_idx].cpu().numpy()
        #print(f"Top 5 important features for node {node_idx}:")
        top_features = np.argsort(feature_importance)[-5:][::-1]
        for i, feat_idx in enumerate(top_features):
            #print(f"  Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
            pass
        
        # Edge importance
        edge_importance = explanation.edge_mask.cpu().numpy()
        #print(f"Number of important edges: {len(edge_importance[edge_importance > 0.5])}")

    # Visualize feature importance for all nodes
    print("\nGenerating feature importance visualization...")

    # Get explanations for all nodes
    all_feature_importance = []
    for i in range(min(100, data.num_nodes)):  # Limit to first 100 nodes for efficiency
        all_feature_importance.append(explanation.node_mask[i].cpu().numpy())

    all_feature_importance = np.array(all_feature_importance)

    # Plot heatmap of feature importance
    plt.figure(figsize=(12, 8))
    plt.imshow(all_feature_importance.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Importance')
    plt.xlabel('Nodes')
    plt.ylabel('Features')
    plt.title('Feature Importance Across Nodes')
    plt.tight_layout()
    plt.savefig(heatmap_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Explanation complete! Heatmap saved as {heatmap_name}")
    
    


if __name__ == "__main__":

    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, 
                                    route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, 
                                    route_classes_bcn, route_classes_nap)
    datalist_thresholded = normalize_and_threshold(datalist, 0.7)
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
    train_loader = create_train_val_dataloader(dataset_train, batch_size=16)
    test_loader, val_loader = create_test_dataloader(dataset_test, batch_size=16)


    # Initialize and train the model
    num_node_features = dataset_train[0].x.shape[1]  # Number of node features
    hidden_channels = 128
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_dropout(num_node_features=num_node_features, hidden_channels=hidden_channels, num_classes=4)  # Adjust parameters as needed
    model.load_state_dict(torch.load('best_gcn_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load your graph data (replace with your actual data loading)
    # Example for Cora dataset - replace with your data
    dataset = test_loader.dataset  # Assuming test_loader is defined
    data = dataset[0].to(device)

    # Initialize GNN explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    # Explain specific nodes
    node_indices = range(data.x.shape[0])  # Choose nodes to explain
    print(f"Number of nodes to explain: {node_indices}")
    print("Generating explanations...")

    # Generate explanation for the node
    explanation = explainer(data.x, data.edge_index)

    for node_idx in node_indices:
        print(f"\nExplaining node {node_idx}:")
    
        # Feature importance (node_mask)
        feature_importance = explanation.node_mask[node_idx].cpu().numpy()
        print(f"Top 5 important features for node {node_idx}:")
        top_features = np.argsort(feature_importance)[-5:][::-1]
        for i, feat_idx in enumerate(top_features):
            print(f"  Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
        
        # Edge importance
        edge_importance = explanation.edge_mask.cpu().numpy()
        print(f"Number of important edges: {len(edge_importance[edge_importance > 0.5])}")

    # Visualize feature importance for all nodes
    print("\nGenerating feature importance visualization...")

    # Get explanations for all nodes
    all_feature_importance = []
    for i in range( data.num_nodes):  # Limit to first 100 nodes for efficiency
        all_feature_importance.append(explanation.node_mask[i].cpu().numpy())

    all_feature_importance = np.array(all_feature_importance)

    # Plot heatmap of feature importance
    plt.figure(figsize=(12, 8))
    plt.imshow(all_feature_importance.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Importance')
    plt.xlabel('Nodes')
    plt.ylabel('Features')
    plt.title('Feature Importance Across Nodes')
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Explanation complete! Heatmap saved as 'feature_importance_heatmap.png'")
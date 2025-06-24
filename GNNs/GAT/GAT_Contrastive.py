
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import os
from torch_geometric.nn import GraphNorm
from torch.nn import TripletMarginLoss

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts


from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
import wandb
import math

import sys
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG")

from Data_Preprocessing.data_cleaning import create_clean_dataset
from Data_Preprocessing.graph_prunning import normalize_and_threshold
from Data_Preprocessing.DataAugmentation import datalist_mixing, datalist_mixing_balanced
from Data_Preprocessing.CreateMultilayerMS import create_multilayer_ms
from Data_Preprocessing.GraphMetricsIndividual import open_features_dict, create_graph_list_individual
from Data_Preprocessing.GraphMetrics import create_graph_list  
from torch_geometric.nn import GATConv
import csv

from GNNs.GAT.models.GAT_model_dropout import GAT
from GNNs.dataloader.GCN_dataloader import create_gcn_dataset, create_train_val_dataloader, create_test_dataloader, separate_train_test
from GNNs.train_test_pipeline.train_test_pipeline import train_gcn_model, test_gcn_model
from GNNs.GCN.explainer.ExplainResults import explain_model

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

# Initialize W&B
wandb.init(
    project="GAT-Training",  # Replace with your project name
    entity="kerasam12",  # Replace with your W&B username or team name
    config={
        "learning_rate": 0.0005,
        "epochs": 5000,
        "batch_size": 16,
        "hidden_channels": 64,
        "num_classes": "dynamic",  # Will be updated dynamically
    }
)

if __name__ == "__main__":
    # Example usage
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, 
                                    route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, 
                                    route_classes_bcn, route_classes_nap)
    datalist_thresholded = normalize_and_threshold(datalist, 0.7)
    datalist_train,datalist_test = separate_train_test(datalist_thresholded, route_features_train, route_features_test)
    print(f"Training samples: {len(datalist_train)}")
    datalist_augmented = datalist_mixing_balanced(datalist_train, [0,1,2,3], 110)

    # 1. Generate graphs from datalist
    graphlist_train = create_graph_list_individual(datalist_augmented)
    graphlist_test = create_graph_list_individual(datalist_test)
    

    # 2. Load features from disk
    features_dict_train = open_features_dict("computed_features", train = True)
    features_dict_test = open_features_dict("computed_featuress", train = False)
    
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
    

    hidden_channels = 32
    # Calculate number of classes by finding unique labels
    unique_labels = set()
    for _, mstype, _, _ in graphlist_train:
        unique_labels.add(mstype)
    num_classes = len(unique_labels)
    
    model = GAT(num_node_features, hidden_channels, num_classes)
    trained_model = train_gcn_model(model, train_loader, val_loader, num_epochs=500, learning_rate = 0.00001)
    
    # Test the model
    test_acc = test_gcn_model(trained_model, test_loader)
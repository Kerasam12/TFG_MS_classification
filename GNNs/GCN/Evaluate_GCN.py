from GCN_features import GCN, create_gcn_dataset, create_train_val_dataloader, create_test_dataloader, test_gcn_model, separate_train_test
from GAT import GAT
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
    train_loader = create_train_val_dataloader(dataset_train, batch_size=64)
    test_loader, val_loader = create_test_dataloader(dataset_test, batch_size=64)
    # Initialize and train the model
    num_node_features = dataset_train[0].x.shape[1]  # Number of node features
    

    hidden_channels = 128
    # Calculate number of classes by finding unique labels
    unique_labels = set()
    for _, mstype, _, _ in graphlist_train:
        unique_labels.add(mstype)
    num_classes = len(unique_labels)

    best_model_path = 'best_gcn_model.pth'
    best_trained_model = GAT(num_node_features, hidden_channels, num_classes)
    best_trained_model.load_state_dict(torch.load(best_model_path))
    best_trained_model = best_trained_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Test the model

    print("\nTest best Model")
    test_acc_best = test_gcn_model(best_trained_model, test_loader)


# Creating and training a GCN model with PyTorch Geometric
from torch_geometric.datasets import Planetoid
import torch.optim as optim
import torch.nn.functional as F
from Simple_GCN_model import Simple_GCN
import numpy as np
from data_cleaning import create_clean_dataset, normalize_and_threshold
from DataAugmentation import datalist_mixing
from CreateMultilayerMS import create_multilayer_ms
from Try_SVM import generate_balanced_train_test


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

# Load a standard benchmark dataset (Cora)
# Create dataset
prunning_threshold = 0.7
mixing_level = [1,2,3,4,5]
datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
datalist = normalize_and_threshold(datalist, prunning_threshold)

print("Total number of dataopoints:",len(datalist))
ms_types = [datapoint[3] for datapoint in datalist]
    
print(f"NO MS: {ms_types.count(-1)}, Percentage: {ms_types.count(-1) / len(ms_types):.2f}")
print(f"RRMS: {ms_types.count(0)}, Percentage: {ms_types.count(0) / len(ms_types):.2f}")
print(f"SPMS: {ms_types.count(1)}, Percentage: {ms_types.count(1) / len(ms_types):.2f}")
print(f"PPMS: {ms_types.count(2)}, Percentage: {ms_types.count(2) / len(ms_types):.2f}")
print(f"MS types: {np.unique(ms_types)} \n")


np.random.shuffle(datalist)

# Split data into training and testing sets
train_percentage = 0.7
train_len = int(len(datalist)*train_percentage)
datalist_train = datalist[:train_len]
mixed_datalist = datalist_mixing(datalist_train, mixing_level, 2500)


datalist_test = datalist[train_len:]



print("datatpoints for training: ",len(datalist_train))
datalist_new = create_multilayer_ms(mixed_datalist)
print("datapoints for test ",len(datalist_test), "\n")
datalist_test = create_multilayer_ms(datalist_test)


X_train, _, y_train, _ = generate_balanced_train_test(datalist_new, test_size=0)
X_test, _, y_test, _ = generate_balanced_train_test(datalist_test, test_size=0)

X_train = np.array([matrix.flatten() for matrix in X_train])
X_test = np.array([matrix.flatten() for matrix in X_test])

# Get unique classes and their counts
classes_train, counts_train = np.unique(y_train, return_counts=True)

# Compute percentages
percentages = (counts_train / len(y_train)) * 100
for cls, perc in zip(classes_train, percentages):
    print(f"Class {cls}: {perc:.2f}%")

    
print("X_train",X_train.shape)
print("y_train",len(y_train))


# Initialize model
model = Simple_GCN(in_channels=dataset.num_features, 
            hidden_channels=16, 
            out_channels=dataset.num_classes)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    # Forward pass
    out = model(data.x, data.edge_index)
    # Calculate loss (only on training nodes)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print training progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
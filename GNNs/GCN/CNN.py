import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts



import sys
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Data_Preprocessing")

from data_cleaning import create_clean_dataset
from graph_prunning import normalize_and_threshold
from torch.utils.data import Dataset, DataLoader
from DataAugmentation import datalist_mixing, datalist_mixing_balanced
from CreateMultilayerMS import create_multilayer_ms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix


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




class MatrixDataset(Dataset):
    def __init__(self, list_matrices):
        self.list_matrices = list_matrices

    def __len__(self):
        return len(self.list_matrices)

    def __getitem__(self, idx):
        item = self.list_matrices[idx]

        # Assume mstype is always the second-to-last item
        mstype = item[3]
        # All items before mstype and edds are matrices
        matrix_FA = item[0]
        matrix_Func = item[1]
        matrix_GM = item[2]


        # Convert the three 76x76 matrices into a single 3x76x76 matrix
        full_matrix = np.stack((matrix_FA, matrix_Func, matrix_GM), axis=0)
        # Convert mstype to indices
        types = {-1: 0, 0: 1, 1: 2, 2: 3}
        mstype = types[mstype]

        
        return full_matrix, mstype


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        # Define the CNN architecture
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(16)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fractional_pool = nn.FractionalMaxPool2d(5,output_ratio=(0.5, 0.5))
        self.fc1 = nn.Linear(1296, 32)  
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Forward pass through the network 
        #x = x.unsqueeze(1)# set a unique channel for CNN processing
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = self.dropout(x)

        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x = self.dropout(x)

        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_bn(x)
        x = self.dropout(x)

        
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_bn(x)


        x = self.max_pool(x)
        x = self.dropout(x)

        x = x.view(batch_size, -1)  # Flatten the tenso        x = self.fc1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def balance_classes(datalist):
    for idx,datapoint in enumerate(datalist):
        mstype = datapoint[1]

    
# Training function
def train(model, loader, optimizer, scheduler,device):
    loss = 0
    model.train()
    
    total_loss = 0
    criterion = nn.CrossEntropyLoss()   
    for matrix, mstype in loader:
        matrix = matrix.type(torch.float32).to(device)
        mstype = mstype.to(device)

        optimizer.zero_grad()
        out = model(matrix)
        
        train_loss = criterion(out, mstype)
        train_loss.backward()
        optimizer.step()

        loss += train_loss.item()
    scheduler.step()    
    # Calculate accuracy for the current batch
    pred = out.argmax(dim=1)
    batch_accuracy = (pred == mstype).sum().item() / mstype.size(0)
    print(f"Batch Accuracy: {batch_accuracy:.4f}")
    
    # Compute the epoch training loss
    loss = loss / len(loader)
    
    return loss

# Testing function
@torch.no_grad()
def test(model, loader, device):
    model.eval()
    
    correct = 0
    for matrix, mstype in loader:
        matrix = matrix.type(torch.float32).to(device)
        mstype = mstype.to(device)
        out = model(matrix)
        pred = out.argmax(dim=1)
        print("pred", pred)
        print("mstype", mstype)
        correct += int((pred == mstype).sum())
    
    return correct / len(loader.dataset)
@torch.no_grad()
def test_with_metrics(model, loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for matrix, mstype in loader:
        matrix = matrix.type(torch.float32).to(device)
        mstype = mstype.to(device)
        out = model(matrix)
        pred = out.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(mstype.cpu().numpy())
        
        # For now, just collect all predictions and labels
        # We'll print the classification matrix at the end
        
    # Create classification matrix (confusion matrix)
    cm = confusion_matrix(all_labels, all_preds)
    print("\n===== Classification Matrix =====")
    print("       | Predicted Classes")
    print("       | Control  HC      RR      SP")
    print("-------+--------------------------------")
    class_names = ['Control', 'HC    ', 'RR    ', 'SP    ']
    for i, row in enumerate(cm):
        print(f"{class_names[i]} | {row[0]:7d} {row[1]:7d} {row[2]:7d} {row[3]:7d}")
            
    print("\n===== Overall Predictions and Labels =====")
    print("All predictions:", all_preds)
    print("All labels:", all_labels)
    

    
     # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # For multi-class classification, we need to specify averaging method
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f'Test Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:')
    print(conf_matrix)
    print("raw predictions: ", all_preds)
    print("raw labels: ", all_labels)
    # Calculate metrics per class
    class_metrics = {}
    for class_label in sorted(set(all_labels)):
        class_accuracy = accuracy_score(all_labels, all_preds, normalize=True, sample_weight=None)
        class_precision = precision_score(all_labels, all_preds, labels=[class_label], average=None, zero_division=0)[0]
        class_recall = recall_score(all_labels, all_preds, labels=[class_label], average=None, zero_division=0)[0]
        class_f1 = f1_score(all_labels, all_preds, labels=[class_label], average=None, zero_division=0)[0]
        class_metrics[class_label] = {
            "Accuracy": class_accuracy,
            "Precision": class_precision,
            "Recall": class_recall,
            "F1 Score": class_f1
        }
        print(f"Class {class_label}: Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1 Score: {class_f1:.4f}")

    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1
    

    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prunning_threshold = 0.7

    # Create dataset
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    datalist = normalize_and_threshold(datalist, prunning_threshold)
    train_percentage = 0.7
    mixing_levels = [1,2,3,4,5]
    train_len = int(len(datalist)*train_percentage)
    datalist_train = datalist[:train_len]
    datalist_test = datalist[train_len:]


    mixed_datalist = datalist_mixing_balanced(datalist_train, mixing_levels, 500)

    mixed_datalist = normalize_and_threshold(mixed_datalist, prunning_threshold)


    print("datatpoints for training: ",len(datalist_train))
    datalist_new = create_multilayer_ms(mixed_datalist)
    print("datapoints for training after mixing: ",len(datalist_new), "\n")
    mstypes_lis = [datapoint[1] for datapoint in datalist_new]
    
    # Get unique classes and their counts
    classes_train, counts_train = np.unique(mstypes_lis, return_counts=True)
    print("Counts train:", counts_train)
    # Compute percentages
    percentages = (counts_train / len(mstypes_lis)) * 100
    for cls, perc in zip(classes_train, percentages):
        print(f"Class {cls}: {perc:.2f}%, ")

    print("datapoints for test ",len(datalist_test), "\n")
    #datalist_test = create_multilayer_ms(datalist_test)


    
    
    dataloader_train = DataLoader(MatrixDataset(mixed_datalist), batch_size=512, shuffle=True)
    dataloader_test = DataLoader(MatrixDataset(datalist_test), batch_size=512, shuffle=True)


    # Example usage
    model = CNN(in_channels=3, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.000001)
    
    # Move model to device
    model.to(device)
    # Training loop
    for epoch in range(5000):  # Number of epochs
        print(f' start training epoch {epoch+1}/{10}')
        loss = train(model, dataloader_train, optimizer,scheduler, device)
        
        print(f'Epoch {epoch+1}, Loss: {loss:.7f}')

    # Testing loop
    accuracy = test(model, dataloader_test, device)
    test_with_metrics(model, dataloader_test, device)
    print(f'Test Accuracy: {accuracy:.4f}')
   



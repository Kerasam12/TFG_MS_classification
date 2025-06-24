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
from torch.nn.modules.utils import _pair, _quadruple

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
        types = {-1: 3, 0: 0, 1: 1, 2: 2}
        mstype = types[mstype]

        
        return full_matrix, mstype


class StochasticPool2d(nn.Module):
    """ Stochastic 2D pooling, where prob(selecting index)~value of the activation
    IM_SIZE should be divisible by 2, not best implementation.  

    based off:
    https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598#file-median_pool-py-L5
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=2, stride=2, padding=0, same=False):
        super(StochasticPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size) # I don't know what this is but it works
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # because multinomial likes to fail on GPU when all values are equal 
        # Try randomly sampling without calling the get_random function a million times
        init_size = x.shape

        # x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        x = x.contiguous().view(-1, 4)
        idx = torch.randint(0, x.shape[1], size=(x.shape[0],)).type(torch.cuda.LongTensor)
        x = x.contiguous().view(-1)
        x = torch.take(x, idx)
        x = x.contiguous().view(init_size[0], init_size[1], int(init_size[2]/2), int(init_size[3]/2))
        return x


# MS-CNN model based on the paper
class MSCNN(nn.Module):
    def __init__(self):
        super(MSCNN, self).__init__()
        
        # Layer definitions according to Table 4 and 5 in the paper
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        
        # Pool Layer 1
        self.pool1 = StochasticPool2d()
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        
        # Pool Layer 2
        self.pool2 = StochasticPool2d()
        
        # Conv Layer 3
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        
        # Conv Layer 4
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        
        # Conv Layer 5
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()
        
        # Pool Layer 3
        self.pool3 = StochasticPool2d()
        
        # Conv Layer 6
        self.conv6 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()
        
        # Conv Layer 7
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        
        # Conv Layer 8
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()
        
        # Conv Layer 9
        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()
        
        # Conv Layer 10
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.relu10 = nn.ReLU()
        
        # Conv Layer 11
        self.conv11 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.relu11 = nn.ReLU()
        
        # Pool Layer 4
        self.pool4 = StochasticPool2d()
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully Connected Layers according to Table 5
        self.fc1 = nn.Linear(64, 32)  # Assuming input size 128x128
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(32, 20)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(20, 10)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(10, 2)  # Binary classification: MS vs Healthy
        
    def forward(self, x):
        # Conv block 1
        x = self.relu1(self.bn1(self.conv1(x)))
        #x = self.pool1(x)
        
        # Conv block 2
        x = self.relu2(self.bn2(self.conv2(x)))
        #x = self.pool2(x)
        
        # Conv block 3-5
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        #x = self.pool3(x)
        
        # Conv block 6-8
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        
        # Conv block 9-11
        x = self.relu9(self.bn9(self.conv9(x)))
        x = self.relu10(self.bn10(self.conv10(x)))
        x = self.relu11(self.bn11(self.conv11(x)))
        #x = self.pool4(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.dropout3(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
def balance_classes(datalist):
    for idx,datapoint in enumerate(datalist):
        mstype = datapoint[1]

    
# Training function
def train(model, loader, optimizer, scheduler,device):
    loss = 0
    model.train()
    
    total_loss = 0
    for matrix, mstype in loader:
        matrix = matrix.type(torch.float32).to(device)
        mstype = mstype.to(device)

        optimizer.zero_grad()
        out = model(matrix)
        
        train_loss = F.cross_entropy(out, mstype)
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
        class_precision = precision_score(all_labels, all_preds, labels=[class_label], average=None, zero_division=0)[0]
        class_recall = recall_score(all_labels, all_preds, labels=[class_label], average=None, zero_division=0)[0]
        class_f1 = f1_score(all_labels, all_preds, labels=[class_label], average=None, zero_division=0)[0]
        class_metrics[class_label] = {
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
    prunning_threshold = 0.0

    # Create dataset
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    datalist = normalize_and_threshold(datalist, prunning_threshold)
    train_percentage = 0.7
    mixing_levels = [1,2,3,4,5]
    train_len = int(len(datalist)*train_percentage)
    datalist_train = datalist[:train_len]
    datalist_test = datalist[train_len:]


    mixed_datalist = datalist_mixing_balanced(datalist_train, mixing_levels, 500)
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


    
    
    dataloader_train = DataLoader(MatrixDataset(mixed_datalist), batch_size=128, shuffle=True)
    dataloader_test = DataLoader(MatrixDataset(datalist_test), batch_size=128, shuffle=False)


    # Example usage
    model = MSCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    
    # Move model to device
    model.to(device)
    # Training loop
    for epoch in range(150):  # Number of epochs
        print(f' start training epoch {epoch+1}/{10}')
        loss = train(model, dataloader_train, optimizer,scheduler, device)
        
        print(f'Epoch {epoch+1}, Loss: {loss:.7f}')

    # Testing loop
    accuracy = test(model, dataloader_test, device)
    test_with_metrics(model, dataloader_test, device)
    print(f'Test Accuracy: {accuracy:.4f}')





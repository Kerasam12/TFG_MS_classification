import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import TripletMarginLoss

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv

def online_batch_ordering(output, labels, device):
    triplet_loss = TripletMarginLoss(margin=0.7)
    total_loss = torch.tensor(0, requires_grad= True,dtype=float).to(device)
    for xq, qlabel in zip(output, labels):
        for xp, plabel in zip(output, labels):
            for xn, nlabel in zip(output, labels):
                if qlabel == plabel and qlabel != nlabel:
                    res = triplet_loss(xq,xp,xn)
                    total_loss += res
    return total_loss/len(labels)

def train_gcn_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, model_path='best_gcn_model.pth', weight=None):
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
    if weight is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:  # Adjust weights as needed weight=torch.tensor([1.75, 9, 20, 4], dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight.to(device))  # Adjust weights as needed weight=torch.tensor([1.75, 9, 20, 4], dtype=torch.float).to(device)
    criterion_contrastive = online_batch_ordering
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=0.000001)
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            classification_loss = criterion(out, data.y)
            contrastive_loss = criterion_contrastive(out, data.y, device)
            loss = classification_loss + contrastive_loss  # Combine losses
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = out.max(1)
            total += data.y.size(0)
            correct += predicted.eq(data.y).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                classification_loss = criterion(out, data.y)
                contrastive_loss = criterion_contrastive(out, data.y, device)
                loss = classification_loss + contrastive_loss  # Combine losses
                val_loss += loss.item()
                _, predicted = out.max(1)
                total += data.y.size(0)
                correct += predicted.eq(data.y).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save the best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
    
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
            out = model(data.x, data.edge_index, data.batch)
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
import random
import pandas as pd 
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

import sys 
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Data_Preprocessing")

from data_cleaning import create_clean_dataset
from CreateMultilayerMS import create_multilayer_ms
from graph_prunning import graph_prunning_threshold, normalize_and_threshold
from DataAugmentation import datalist_mixing, datalist_mixing_balanced



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

def generate_train_test(datalist):
    """
    Return the list of matrices for training classifyer in the X list.
    Return the list of types of each matrix in the list y. 
    """
    X = [] # List containing the matrix used as data for training the classifyer
    y = [] # List containing the classes of each matrix
    for datapoint in datalist:
        X.append(datapoint[0])
        y.append(datapoint[1])
        
    return X,y

def generate_balanced_train_test(datalist, test_size=0.3):
    """
    Balance dataset by including samples of each class in training and testing sets."""
    train_size = 1 - test_size
    types_dict = {0:[], 1:[], 2:[], -1:[]}
    for datapoint in datalist:
        matrix = datapoint[0]
        mstype = datapoint[1]
        
        types_dict[mstype].append(matrix)

    len_per_class = [len(types_dict[key]) for key in types_dict.keys()]
    print("len_per_class",len_per_class)


    len_train_0 = int(len(types_dict[0]) * train_size)
    len_train_1 = int(len(types_dict[1]) * train_size)
    len_train_2 = int(len(types_dict[2]) * train_size)
    len_train_neg1 = int(len(types_dict[-1]) * train_size)

    len_test_0 = len(types_dict[0]) - len_train_0
    len_test_1 = len(types_dict[1]) - len_train_1
    len_test_2 = len(types_dict[2]) - len_train_2
    len_test_neg1 = len(types_dict[-1]) - len_train_neg1



    X_train = types_dict[0][: len_train_0] + types_dict[1][: len_train_1] + types_dict[2][: len_train_2] + types_dict[-1][: len_train_neg1]
    print("X_train",len(X_train), X_train[0].shape)
    y_train = [1] * len_train_0 + [2] * len_train_1 + [3] * len_train_2 + [0] * len_train_neg1
    X_test = types_dict[0][len_train_0:] + types_dict[1][len_train_1:] + types_dict[2][len_train_2:] + types_dict[-1][len_train_neg1:]
    y_test = [1] * len_test_0 + [2] * len_test_1 + [3] * len_test_2 + [0] * len_test_neg1

    return X_train, X_test, y_train, y_test

def generate_binary_balanced_train_test(datalist, test_size=0.3):
    """
    Balance dataset by including samples of each class in training and testing sets."""

    train_size = 1 - test_size
    types_dict = {0:[], 1:[], 2:[], -1:[]}
    
    for datapoint in datalist:
        matrix = datapoint[0]
        mstype = datapoint[1]




        edds = datapoint[2]
        types_dict[mstype].append(matrix)

    len_per_class = [len(types_dict[key]) for key in types_dict.keys()]
    print("len_per_class",len_per_class)
    min_binary_len = min(len_per_class[-1], len_per_class[0])
    

    len_train_0 = min_binary_len#int(len(types_dict[0]) * train_size)
    len_train_1 = 0#int(len(types_dict[1]) * train_size)
    len_train_2 = 0#int(len(types_dict[2]) * train_size)
    len_train_neg1 = min_binary_len#int(len(types_dict[-1]) * train_size)

    len_test_0 = len(types_dict[0]) - len_train_0
    len_test_1 = len(types_dict[1]) - len_train_1
    len_test_2 = len(types_dict[2]) - len_train_2
    len_test_neg1 = len(types_dict[-1]) - len_train_neg1


    



    X_train = types_dict[0][: len_train_0] + types_dict[1][: len_train_1] + types_dict[2][: len_train_2] + types_dict[-1][: len_train_neg1]
    print("X_train",len(X_train), X_train[0].shape)
    y_train = [1] * len_train_0 + [1] * len_train_1 + [1] * len_train_2 + [0] * len_train_neg1
    X_test = types_dict[0][len_train_0:] + types_dict[1][len_train_1:] + types_dict[2][len_train_2:] + types_dict[-1][len_train_neg1:]
    y_test = [1] * len_test_0 + [1] * len_test_1 + [1] * len_test_2 + [0] * len_test_neg1
   
    return X_train, X_test, y_train, y_test

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




if __name__ == "__main__":
    # Define parameters
    kernels = ['linear']
    thresholds = [0.7]
    mixing_levels = [0,1,2, 3, 4]
    matrix_type = ["FA","fMRI", "GM"]



    # Create dataset
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)

    datalist_train, datalist_test = generate_train_test_datalist(datalist, train_ratio = 0.7)
    
    for kernel in kernels:
        results = []
        
        for prunning_threshold in thresholds:
            # Normalize and trheshold
            datalist_train = normalize_and_threshold(datalist_train, prunning_threshold)
            datalist_test = normalize_and_threshold(datalist_test, prunning_threshold)
            
            mix = [3]
            print("Mix: ",mix)
            print("Mixing levels used: ",mix)
            mixed_datalist = datalist_train#datalist_mixing_balanced(datalist_train, mix, 5000)
            print("Mixed data points: ",len(mixed_datalist))


            print("datatpoints for training FA:  ",len(datalist_train))
            datalist_single_matrix_train_FA = [(datapoint[0],datapoint[3]) for datapoint in mixed_datalist]
            print("datapoints for test FA",len(datalist_test), "\n")
            datalist_single_matrix_test_FA = [(datapoint[0],datapoint[3]) for datapoint in datalist_test]

            print("datatpoints for training fMRI:  ",len(datalist_train))
            datalist_single_matrix_train_fMRI = [(datapoint[1],datapoint[3]) for datapoint in mixed_datalist]
            print("datapoints for test fMRI",len(datalist_test), "\n")
            datalist_single_matrix_test_fMRI = [(datapoint[1],datapoint[3]) for datapoint in datalist_test]

            print("datatpoints for training GM:  ",len(datalist_train))
            datalist_single_matrix_train_GM = [(datapoint[2],datapoint[3]) for datapoint in mixed_datalist]
            print("datapoints for test GM",len(datalist_test), "\n")
            datalist_single_matrix_test_GM = [(datapoint[2],datapoint[3]) for datapoint in datalist_test]

            
            X_train_FA, y_train = generate_train_test(datalist_single_matrix_train_FA)
            X_test_FA, y_test= generate_train_test(datalist_single_matrix_test_FA)

            X_train_fMRI, y_train = generate_train_test(datalist_single_matrix_train_fMRI)
            X_test_fMRI, y_test= generate_train_test(datalist_single_matrix_test_fMRI)

            X_train_GM, y_train = generate_train_test(datalist_single_matrix_train_GM)
            X_test_GM, y_test= generate_train_test(datalist_single_matrix_test_GM)


            X_train_FA = np.array([matrix.flatten() for matrix in X_train_FA])
            X_test_FA = np.array([matrix.flatten() for matrix in X_test_FA])

            X_train_fMRI = np.array([matrix.flatten() for matrix in X_train_fMRI])
            X_test_fMRI = np.array([matrix.flatten() for matrix in X_test_fMRI])

            X_train_GM = np.array([matrix.flatten() for matrix in X_train_GM])
            X_test_GM = np.array([matrix.flatten() for matrix in X_test_GM])
         
            # Get unique classes and their counts
            classes_train, counts_train = np.unique(y_train, return_counts=True)

            # Compute percentages
            percentages = (counts_train / len(y_train)) * 100
            for cls, perc in zip(classes_train, percentages):
                print(f"Class {cls}: {perc:.2f}%")

                
            print("X_train",X_train_FA.shape)
            print("y_train",len(y_train))
            

            # Train SVM
            clf_FA = svm.SVC(kernel=kernel, probability=True, class_weight='balanced')#, class_weight={0:1/0.26,1: 1/0.58,2:1/0.11, 3:1/0.02}
            clf_fMRI = svm.SVC(kernel=kernel,probability=True, class_weight='balanced')
            clf_GM = svm.SVC(kernel=kernel, probability=True, class_weight='balanced')

            y_pred_FA = clf_FA.fit(X_train_FA, y_train).predict(X_test_FA)
            y_pred_fMRI = clf_fMRI.fit(X_train_fMRI, y_train).predict(X_test_fMRI)
            y_pred_GM = clf_GM.fit(X_train_GM, y_train).predict(X_test_GM)

            y_probs_FA = clf_FA.predict_proba(X_test_FA)
            y_probs_fMRI = clf_fMRI.predict_proba(X_test_fMRI)
            y_probs_GM = clf_GM.predict_proba(X_test_GM)

            y_probs = np.array([y_probs_FA]) + np.array([y_probs_fMRI]) + np.array([y_probs_GM])


            y_pred = []

            conversion_dict = {0:-1, 1: 0, 2: 1, 3: 2}



            for idx,(y_FA, y_fMRI, y_GM) in enumerate(zip(y_pred_FA, y_pred_fMRI, y_pred_GM)):
                if y_FA != y_fMRI and y_FA != y_GM and y_fMRI != y_GM:
                    pred_class_idx = np.argmax(y_probs [0][idx])
                    pred_class = conversion_dict[pred_class_idx]
                    y_pred.append(pred_class)
                    print("tie", idx)

                else:
                    most_frequent = Counter([y_FA,y_fMRI, y_GM]).most_common(1)[0][0]
                    y_pred.append(most_frequent)

            

            print("y_pred",y_pred)
            print("y_test",y_test)

            conf_matrix = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(conf_matrix)
            # Calculate average metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision_avg = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_avg = recall_score(y_test, y_pred, average='macro', zero_division=0)

            print("\n===== AVERAGE METRICS =====")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro Precision: {precision_avg:.4f}")
            print(f"Macro Recall: {recall_avg:.4f}")


            # Calculate per-class metrics
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)

            # Calculate per-class accuracy
            conf_matrix = confusion_matrix(y_test, y_pred)
            per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

            print("\n===== PER-CLASS METRICS =====", per_class_accuracy)
            print("class", range(len(np.unique(y_test))))
            
            metrics_df = pd.DataFrame({
                'Model': "SVM Voting Classifier",
                'Class': range(len(np.unique(y_test))),
                'Accuracy': per_class_accuracy,
                'Precision': precision_per_class,
                'Recall': recall_per_class
            })

            metrics_df.to_csv('metrics_svm_no_cross.csv', index=False)

            print(metrics_df)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from data_cleaning import create_clean_dataset
from CreateMultilayerMS import create_multilayer_ms
import random
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from graph_prunning import graph_prunning_threshold, normalize_and_threshold
from DataAugmentation import datalist_mixing, datalist_mixing_balanced
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
import pandas as pd 
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter


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
    
    thresholds = [0.5, 0.7, 0.8, 0.9]
    mixing_levels = [0,1, 2, 3, 4]


    # Create dataset
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    results = []
    

    print("Total number of dataopoints:",len(datalist))
    ms_types = [datapoint[3] for datapoint in datalist]
        
    print(f"NO MS: {ms_types.count(-1)}, Percentage: {ms_types.count(-1) / len(ms_types):.2f}")
    print(f"RRMS: {ms_types.count(0)}, Percentage: {ms_types.count(0) / len(ms_types):.2f}")
    print(f"SPMS: {ms_types.count(1)}, Percentage: {ms_types.count(1) / len(ms_types):.2f}")
    print(f"PPMS: {ms_types.count(2)}, Percentage: {ms_types.count(2) / len(ms_types):.2f}")
    print(f"MS types: {np.unique(ms_types)} \n")

    datalist_train, datalist_test = generate_train_test_datalist(datalist, train_ratio = 0.7)
    
    for prunning_threshold in thresholds:
        # Normalize and trheshold
        datalist_train = normalize_and_threshold(datalist_train, prunning_threshold)
        datalist_test = normalize_and_threshold(datalist_test, prunning_threshold)


        for mix in mixing_levels:

            print("Mix: ",mix)
            print("Mixing levels used: ",mixing_levels[:mix+1])
            mixed_datalist = datalist_mixing_balanced(datalist_train, mixing_levels[:mix+1], 5000)
            print("Mixed data points: ",len(mixed_datalist))

            print("datatpoints for training: ",len(datalist_train))
            datalist_new = create_multilayer_ms(mixed_datalist)
            print("datapoints for test ",len(datalist_test), "\n")
            print("datapoints for training ",datalist_test[0][0], "\n")
            datalist_test1 = create_multilayer_ms(datalist_test)
            

            X_train, _, y_train, _ = generate_balanced_train_test(datalist_new, test_size=0)
            X_test, _, y_test, _ = generate_balanced_train_test(datalist_test1, test_size=0)

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
            

            # Train SVM
            clf = tree.DecisionTreeClassifier(max_depth = 8)#, class_weight={0:1/0.26,1: 1/0.58,2:1/0.11, 3:1/0.02}

            y_pred = clf.fit(X_train, y_train).predict(X_test)
            print("y_pred",y_pred)
            print("y_test",y_test)

            scores = cross_validate(clf, X_test, y_test, cv=5, scoring=('accuracy', 'recall_weighted', 'f1_weighted', 'precision_weighted' ))
            print( "Scores",scores.keys())
            print("\n")
            print("Accuracy:", scores['test_accuracy'].mean())
            print("Full Accuracy", scores['test_accuracy'])
            print("Recall:", scores['test_recall_weighted'].mean())
            print("F1:", scores['test_f1_weighted'].mean())
            print("Precision:", scores['test_precision_weighted'].mean())

            conf_matrix = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(conf_matrix)


            # Append results
            results.append({
                "Threshold": prunning_threshold,
                "Mixing Level": mix,
                "Mean Accuracy": scores['test_accuracy'].mean(),
                "Mean Precision": scores['test_precision_weighted'].mean(),
                "Mean Recall": scores['test_recall_weighted'].mean(),
                "Mean F1 Score": scores['test_f1_weighted'].mean(),
                "No MS Accuracy": scores['test_accuracy'][0],
                "RRMS Accuracy": scores['test_accuracy'][1],
                "SPMS Accuracy": scores['test_accuracy'][2],
                "PPMS Accuracy": scores['test_accuracy'][3],
                "No MS Precision": scores['test_precision_weighted'][0],
                "RRMS Precision": scores['test_precision_weighted'][1],
                "SPMS Precision": scores['test_precision_weighted'][2],
                "PPMS Precision": scores['test_precision_weighted'][3],
                "No MS Recall": scores['test_recall_weighted'][0],
                "RRMS Recall": scores['test_recall_weighted'][1],
                "SPMS Recall": scores['test_recall_weighted'][2],
                "PPMS Recall": scores['test_recall_weighted'][3],

                
                #"Confusion Matrix": conf_matrix
            })
    # Convert to DataFrame and save as CSV
    df_results = pd.DataFrame(results)
    filename = f"trees_full_results.csv"
    df_results.to_csv(filename, index=False)
    print(f"Saved results for tree: -> {filename}")
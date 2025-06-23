import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import random
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


import sys 
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Data_Preprocessing")

from data_cleaning import create_clean_dataset
from CreateMultilayerMS import create_multilayer_ms
from graph_prunning import graph_prunning_threshold, normalize_and_threshold
from DataAugmentation import datalist_mixing


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
        edds = datapoint[2]
        types_dict[mstype].append(matrix)


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




if __name__ == "__main__":
    #datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)

    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)

    prunning_threshold = 0.7
    datalist = normalize_and_threshold(datalist, prunning_threshold)
        
    #datalist = create_multilayer_ms(datalist)
    #mixed_datalist = datalist_mixing(datalist, 0.5, 1000)

    #img_hat_list = graph_prunning_DCT(multilayer_list, tau=2, STEP=1)
    #datalist = graph_prunning_threshold(datalist, 0.7)

    print("datalist",len(datalist))
    ms_types = [datapoint[3] for datapoint in datalist]
        
    print(f"NO MS: {ms_types.count(-1)}, Percentage: {ms_types.count(-1) / len(ms_types):.2f}")
    print(f"RRMS: {ms_types.count(0)}, Percentage: {ms_types.count(0) / len(ms_types):.2f}")
    print(f"SPMS: {ms_types.count(1)}, Percentage: {ms_types.count(1) / len(ms_types):.2f}")
    print(f"PPMS: {ms_types.count(2)}, Percentage: {ms_types.count(2) / len(ms_types):.2f}")
    print(f"MS types: {np.unique(ms_types)}")


    random.shuffle(datalist)

    # Create a list of feature arrays and a list of corresponding class labels
    fa_array_list = [datapoint[0] for datapoint in datalist]
    fmri_array_list = [datapoint[1] for datapoint in datalist]
    gm_array_list = [datapoint[2] for datapoint in datalist]

    fa_class_list = [datapoint[3] for datapoint in datalist]

    dict_conversion = {-1:0, 0:1, 1:2, 2:3}
    #fa_class_list = [dict_conversion[ms_type] for ms_type in fa_class_list]



    #array_list = np.concatenate((fa_array_list, fmri_array_list, gm_array_list), axis=1)
    # Create feature vectors for SVM (flatten the image matrices)
    X = np.array([matrix[0].flatten() for matrix in datalist])
    y = np.array([datapoint[3] for datapoint in datalist])

    print("Xshape",X.shape)
    # Split data into train and test sets
    #X_train, X_test = X[:-30], X[-30:]
    #y_train, y_test = y[:-30], y[-30:]

    #X_train, X_test, y_train, y_test = generate_balanced_train_test(datalist, test_size=0.2)
    datalist_train = datalist[:int(len(datalist)*0.5)]
    datalist_test = datalist[int(len(datalist)*0.5):]
    mixed_datalist = datalist_mixing(datalist_train, 0.5, 10000)
    print("datalist_train",len(datalist_train))
    datalist_new = create_multilayer_ms(mixed_datalist)
    print("datalist_test",len(datalist_test))
    datalist_test = create_multilayer_ms(datalist_test)
    

    X_train, _, y_train, _ = generate_binary_balanced_train_test(datalist_new, test_size=0)
    X_test, _, y_test, _ = generate_binary_balanced_train_test(datalist_test, test_size=0)

    X_train = np.array([matrix.flatten() for matrix in X_train])
    X_test = np.array([matrix.flatten() for matrix in X_test])


    sm = SMOTE(random_state=42, sampling_strategy='not majority')
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    # Get unique classes and their counts
    classes_train, counts_train = np.unique(y_train, return_counts=True)

    # Compute percentages
    percentages = (counts_train / len(y_train)) * 100
    for cls, perc in zip(classes_train, percentages):
        print(f"Class {cls}: {perc:.2f}%")

    # Get unique classes and their counts
    classes_train_smote, counts_train_smote = np.unique(y_train_smote, return_counts=True)

    # Compute percentages
    percentages_smote = (counts_train_smote / len(y_train_smote)) * 100
    for cls, perc in zip(classes_train_smote, percentages_smote):
        print(f"Class {cls}: {perc:.2f}%")
        
    print("X_train",X_train.shape)
    print("y_train",len(y_train))
    print("X_train_smote",X_train_smote.shape)
    print("y_train_smote",len(y_train_smote))
    # Train SVM

    clf_smote = svm.SVC(kernel="rbf")

    
    y_pred_smote = clf_smote.fit(X_train_smote, y_train_smote).predict(X_test)
    print("y_test",y_test)

    scores_smote = cross_validate(clf_smote, X_test, y_test, cv=5, scoring=('accuracy', 'recall_weighted', 'f1_weighted'))

    print("Accuracy SMOTE:", scores_smote['test_accuracy'].mean())
    print("Recall SMOTE:", scores_smote['test_recall_weighted'].mean())
    print("F1 SMOTE:", scores_smote['test_f1_weighted'].mean())

    conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
    print("\nConfusion Matrix SMOTE:")
    print(conf_matrix_smote)


    '''clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)
    print("Ground truth",y)
    print("Predictions:", predictions)

    # Compute evaluation metrics
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')'''
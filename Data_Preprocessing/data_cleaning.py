import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
import re
import csv 



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



def extract_names_csv(route_FA, route_Func, route_GM):
    ''' This function return the name of all the files in .csv format in the folders of the different data routes '''

    fa_names = os.listdir(route_FA)
    func_names = os.listdir(route_Func)
    gm_names = os.listdir(route_GM)

    fa_names_csv = [name for name in fa_names if name[-4:] == ".csv"]
    func_names_csv = [name for name in func_names if name[-4:] == ".csv"]
    gm_names_csv = [name for name in gm_names if name[-4:] == ".csv"]
    
    #Check if the names are sorted, if not sort them
    list_sorted = 1 
    for fa, func, gm in zip(fa_names_csv, func_names_csv, gm_names_csv):
        if fa[:8] != func[:8] or fa[:8] != gm[:8]:
            list_sorted = 0
            print("The names are not sorted")
            break
    if list_sorted == 0:
        fa_names_csv, func_names_csv, gm_names_csv = sort_names(fa_names_csv, func_names_csv, gm_names_csv)
        list_sorted = 1 
        for fa, func, gm in zip(fa_names_csv, func_names_csv, gm_names_csv):
            if fa[:8] != func[:8] or fa[:8] != gm[:8]:
                list_sorted = 0
                print("The names are not sorted AGAIN")

    print("FA names sorted", len(fa_names_csv), len(set(fa_names_csv)), "\n")
    print("Func names sorted", len(func_names_csv), len(set(func_names_csv)), "\n")
    print("GM names sorted", len(gm_names_csv), len(set(gm_names_csv)), "\n")
    return fa_names_csv, func_names_csv, gm_names_csv

def sort_names(fa_names_csv, func_names_csv, gm_names_csv):
    ''' This function sorts the names of the files in the folders, so we can have the same order in all the folders '''

    # Extract the first 8 letters of each name in each list
    prefixes1 = {name[:8] for name in fa_names_csv}
    prefixes2 = {name[:8] for name in func_names_csv}
    prefixes3 = {name[:8] for name in gm_names_csv}

    # Find common prefixes in all three lists
    common_prefixes = prefixes1 & prefixes2 & prefixes3

    # Filter lists to keep only names that have common prefixes
    fa_names_csv = sorted([name for name in fa_names_csv if name[:8] in common_prefixes])
    func_names_csv = sorted([name for name in func_names_csv if name[:8] in common_prefixes])
    gm_names_csv = sorted([name for name in gm_names_csv if name[:8] in common_prefixes])

    return fa_names_csv, func_names_csv, gm_names_csv

def extract_data_bcn(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_classes_bcn):
    ''' This function reads all the folders with data, returning the matrices they contain, together with their ground truth labels. 
    There's 3 folders containing data in form of matrices and 1 excel containing the ground truth data

    INPUT


    Disclaimer: In all the folders every datapoint(matrix) is in 3 different formats (1- .csv, 2- .mat, 3- .png) - For our purpouse we will only keep the .csv format
    
    FOLDERS:
        1. FA/DTI: Folder containing the matrices referent to the FA or DTI 
        2. Func/rsFMRI: Folder containing the matrices referent to the Functional analyisis of the brain or also called rsFMRI
        3. GM: Folder containing the matrices referent to the Gray Matter connections of the brain 

    EXCEL FILE:
    This file contains multiple data referent to the patients, but we are only interested in 3 columns
        1. mstype: Referent to the Multiple sclerosis type of the patient. This type can be 0: RRMS, 1: SPMS i 2: PPMS, -1: Don't have MS
        2. edds: expanded disability status scale. Continuous values representing the state of the multiple sclerosis patient. 0 being with no disease, and 10 being death
        3. name: The name of the patient. This name is used to match the data with the excel file. The name is the first 7 or 8 letters of the file name in the folders

    OUTPUT
 
    list of length = num of patients
    In each position we find a tuple with 5 values of a single subject:
        1. FA matrix
        2. Func matrix
        3. GM matrix
        4. mstype value
        5. edds value
        6. name/idx of the subject 

    final_matrix = [(FA_matrix_patient1, Func_matrix_patient1, GM_matrix_patient1, mstype_patient1, edds_patient1, name1), (...), ..., (FA_matrix_patientN, Func_matrix_patientN, GM_matrix_patientN, mstype_patientN, edds_patientN, nameN), ]

    '''

    #Extract all the names in .csv format of every folder 
    fa_names_patients_bcn_csv, func_names_patients_bcn_csv, gm_names_patients_bcn_csv = extract_names_csv(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn)
    fa_names_controls_bcn_csv, func_names_controls_bcn_csv, gm_names_controls_bcn_csv = extract_names_csv(route_controls_FA_bcn, route_controls_Func_bcn, route_controls_GM_bcn)
    print("FA names patients", len(fa_names_patients_bcn_csv), len(set(fa_names_patients_bcn_csv)), "\n")
    print("Func names patients", len(fa_names_controls_bcn_csv), len(set(fa_names_controls_bcn_csv)))

    #Join controls with patients
    fa_names_bcn_csv = fa_names_patients_bcn_csv + fa_names_controls_bcn_csv
    func_names_bcn_csv = func_names_patients_bcn_csv + func_names_controls_bcn_csv
    gm_names_bcn_csv = gm_names_patients_bcn_csv + gm_names_controls_bcn_csv
   
    #Read the excel file with patient data and convert it into a pandas dataframe
    general_info = pd.read_excel(route_classes_bcn)
    general_info_df = pd.DataFrame(general_info)
    
    #Print the number of subjects to check we have the same number of every type 
    print("Num FA_subjects_BCN", len(fa_names_bcn_csv))
    print("Num Func_subjects_BCN", len(func_names_bcn_csv))
    print("Num GM_subjects_BCN", len(gm_names_bcn_csv))

    #Read the first matrix in csv to know the dimensions
    
    FA = pd.read_csv(route_patients_FA_bcn + fa_names_bcn_csv[0], index_col=None, header=None)# read the csv file
    fa_array = np.array(FA)#[:,1:] 
    Func = pd.read_csv(route_patients_Func_bcn + func_names_bcn_csv[0], index_col=None, header=None)
    func_array = np.array(Func)#[:,1:]
    GM = pd.read_csv(route_patients_GM_bcn + gm_names_bcn_csv[0], index_col=None, header=None)
    gm_array = np.array(GM)#[:,1:]

    print("FA shape", fa_array.shape)
    print("Func shape", func_array.shape)
    print("GM shape", gm_array.shape)

    #Create a list where we can store all the csv in a matrix format
    data_list_bcn = list()
    dataset_name = "BCN"

    #Iterate through every csv file, convert it into numpy array and add it into a general "list" with all other matrices
    for fa, func, gm in zip(fa_names_bcn_csv, func_names_bcn_csv, gm_names_bcn_csv):
        #Check if all the datapoint names are the same
        assert fa[:8] == func[:8] and fa[:8] == gm[:8], "The names are not sorted"
        #If the name of the file starts with "c" it means it's a control, if it starts with "F" it means it's a patient
        if fa[0] == "c" or fa[0] == "s":
            if fa[0] == "c":
                name = fa[:9]
            elif fa[0] == "s":
                name = fa[:7]

            fa_class = general_info_df[general_info_df['id'] == name]
            edds = general_info_df[general_info_df['id'] == name]
            #Read the csv files
            FA = pd.read_csv(route_controls_FA_bcn + fa, index_col=None, header=None)# read the csv file 
            Func = pd.read_csv(route_controls_Func_bcn + func, index_col=None, header=None)
            GM = pd.read_csv(route_controls_GM_bcn + gm, index_col=None, header=None)

            #Transform the csv to np array and drop the index column. Add the resulting array in the tuple 
            datapoint = (np.array(FA), np.array(Func), np.array(GM), -1, edds, name, dataset_name)
            # add the resulting datapoints  list
            data_list_bcn.append(datapoint)
        
        if fa[0] == "F":
            name = fa[:7]
        
        if fa[0] == "0" or fa[0] == "1":
            name = fa[:8]

        if fa[0] == "F" or fa[0] == "0" or fa[0] == "1":
  
            fa_class = general_info_df[general_info_df['id'] == name]
            edds = general_info_df[general_info_df['id'] == name]
            if len(edds) == 0:
                edds = -1 #If the edds value is not present, we will assign it a value of -1
            else:
                edds = edds["edss"].values[0]
            if len(fa_class) != 0:
                fa_class = fa_class["mstype"].values[0]
                

                #Read the csv files
                FA = pd.read_csv(route_patients_FA_bcn + fa, index_col=None, header=None)# read the csv file 
                Func = pd.read_csv(route_patients_Func_bcn + func, index_col=None, header=None)
                GM = pd.read_csv(route_patients_GM_bcn + gm, index_col=None, header=None)
                
                #Transform the csv to np array and drop the index column. Add the resulting array in the tuple 
                datapoint = (np.array(FA), np.array(Func), np.array(GM), fa_class, edds, name,dataset_name)
                # add the resulting datapoints  list
                data_list_bcn.append(datapoint)
        else:
            "The patient", name, "is not in the excel file"
            pass

    return data_list_bcn



def extract_data_nap(route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_nap):
    ''' This function reads all the folders with data, returning the matrices they contain, together with their ground truth labels. 
    There's 3 folders containing data in form of matrices and 1 excel containing the ground truth data

    INPUT


    Disclaimer: In all the folders every datapoint(matrix) is in 3 different formats (1- .csv, 2- .mat, 3- .png) - For our purpouse we will only keep the .csv format
    
    FOLDERS:
        1. FA/DTI: Folder containing the matrices referent to the FA or DTI 
        2. Func/rsFMRI: Folder containing the matrices referent to the Functional analyisis of the brain or also called rsFMRI
        3. GM: Folder containing the matrices referent to the Gray Matter connections of the brain 

    EXCEL FILE:
    This file contains multiple data referent to the patients, but we are only interested in 2 columns
        1. mstype: Referent to the Multiple sclerosis type of the patient. This type can be 0: RRMS, 1: SPMS i 2: PPMS, -1: Don't have MS
        2. edds: expanded disability status scale. Continuous values representing the state of the multiple sclerosis patient. 0 being with no disease, and 10 being death
        3. name: The name of the patient. This name is used to match the data with the excel file. The name is the first 8 letters of the file name in the folders

    OUTPUT
 
    list of length = num of patients
    In each position we find a tuple with 5 values of a single subject:
        1. FA matrix
        2. Func matrix
        3. GM matrix
        4. mstype value
        5. edds value
        6. name/idx of the subject

    final_matrix = [(FA_matrix_patient1, Func_matrix_patient1, GM_matrix_patient1, mstype_patient1, edds_patient1), (...), ..., (FA_matrix_patientN, Func_matrix_patientN, GM_matrix_patientN, mstype_patientN, edds_patientN), ]

    '''

    #Extract all the names in .csv format of every folder 
    fa_names_nap_csv, func_names_nap_csv, gm_names_nap_csv = extract_names_csv(route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap)
    
    #Read the excel file with patient data and convert it into a pandas dataframe
    general_info_nap = pd.read_excel(route_classes_nap)
    general_info_nap_df = pd.DataFrame(general_info_nap)
    
    #Print the number of subjects to check we have the same number of every type 
    print("Num FA_subjects_NAP", len(fa_names_nap_csv))
    print("Num Func_subjects_NAP", len(func_names_nap_csv))
    print("Num GM_subjects_NAP", len(gm_names_nap_csv))

    #Read the first matrix in csv to know the dimensions
    FA = pd.read_csv(route_patients_FA_nap + fa_names_nap_csv[0], index_col=None, header=None)
    fa_array = np.array(FA)#[:,1:]#don't keep the first dimension as it's the index in the matrix 
    Func = pd.read_csv(route_patients_Func_nap + func_names_nap_csv[0], index_col=None, header=None)
    func_array = np.array(Func)#[:,1:]
    GM = pd.read_csv(route_patients_GM_nap + gm_names_nap_csv[0], index_col=None, header=None)
    gm_array = np.array(GM)#[:,1:]

    print("FA shape", fa_array.shape)
    print("Func shape", func_array.shape)
    print("GM shape", gm_array.shape)
    print("\n")

    #Create a numpy array where we can store all the csv in a matrix format
    data_list_nap = list()

    ms_types_dict = {"RR": 0, "SP": 1, "PP": 2, "CONTROL": -1}

    #Iterate through every csv file, convert it into numpy array and add it into a general "list" with all other matrices
    for fa, func, gm in zip(fa_names_nap_csv, func_names_nap_csv, gm_names_nap_csv):
        #Check if all the datapoint names are the same
        assert fa[:8] == func[:8] and fa[:8] == gm[:8], "The names are not sorted"
        
        name = fa[:8]

        fa_class_binary = general_info_nap_df[general_info_nap_df['ID'] == name]["GROUP"].values[0]
        if fa_class_binary:
            fa_class_ms = general_info_nap_df[general_info_nap_df['ID'] == name]["GROUP_MS"].values[0]
            fa_class = ms_types_dict[fa_class_ms]
            edds = general_info_nap_df[general_info_nap_df['ID'] == name]["EDSS"].values[0]

        else:
            fa_class = -1
            edds = -1

    
        #Read the csv files
        FA = pd.read_csv(route_patients_FA_nap + fa, index_col=None, header=None)# read the csv file 
        Func = pd.read_csv(route_patients_Func_nap + func, index_col=None, header=None)
        GM = pd.read_csv(route_patients_GM_nap + gm, index_col=None, header=None)

        dataset_name = "NAP"
        #Transform the csv to np array and drop the index column. Add the resulting array in the tuple 
        datapoint = (np.array(FA), np.array(Func), np.array(GM), fa_class, edds, name,dataset_name)
        # add the resulting datapoints  list
        data_list_nap.append(datapoint)
        

    return data_list_nap

def extract_features_bcn(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_classes_bcn, route_features_controls_bcn, route_features_patients_bcn):
    ''' This function reads all the folders with data, returning the matrices they contain, together with their ground truth labels. 
    There's 3 folders containing data in form of matrices and 1 excel containing the ground truth data

    INPUT


    Disclaimer: In all the folders every datapoint(matrix) is in 3 different formats (1- .csv, 2- .mat, 3- .png) - For our purpouse we will only keep the .csv format
    
    FOLDERS:
        1. FA/DTI: Folder containing the matrices referent to the FA or DTI 
        2. Func/rsFMRI: Folder containing the matrices referent to the Functional analyisis of the brain or also called rsFMRI
        3. GM: Folder containing the matrices referent to the Gray Matter connections of the brain 

    EXCEL FILE:
    This file contains multiple data referent to the patients, but we are only interested in 2 columns
        1. mstype: Referent to the Multiple sclerosis type of the patient. This type can be 0: RRMS, 1: SPMS i 2: PPMS, -1: Don't have MS
        2. edds: expanded disability status scale. Continuous values representing the state of the multiple sclerosis patient. 0 being with no disease, and 10 being death

    EXEL WITH FEATURES:
    This file contains multiple data referent to the patients. We will extract all the features for every patient 
        

    OUTPUT
 
    list of length = num of patients
    In each position we find a tuple with 6 values of a single subject:
        1. FA matrix
        2. Func matrix
        3. GM matrix
        4. mstype value
        5. edds value
        6. features value

    final_matrix = [(FA_matrix_patient1, Func_matrix_patient1, GM_matrix_patient1, mstype_patient1, edds_patient1, features_values1), (...), ..., (FA_matrix_patientN, Func_matrix_patientN, GM_matrix_patientN, mstype_patientN, edds_patientN, features_valuesN), ]

    '''

    #Extract all the names in .csv format of every folder 
    fa_names_patients_bcn_csv, func_names_patients_bcn_csv, gm_names_patients_bcn_csv = extract_names_csv(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn)
    fa_names_controls_bcn_csv, func_names_controls_bcn_csv, gm_names_controls_bcn_csv = extract_names_csv(route_controls_FA_bcn, route_controls_Func_bcn, route_controls_GM_bcn)
    
    #Join controls with patients
    fa_names_bcn_csv = fa_names_patients_bcn_csv + fa_names_controls_bcn_csv
    func_names_bcn_csv = func_names_patients_bcn_csv + func_names_controls_bcn_csv
    gm_names_bcn_csv = gm_names_patients_bcn_csv + gm_names_controls_bcn_csv
   
    #Read the excel file with patient data and convert it into a pandas dataframe
    general_info = pd.read_excel(route_classes_bcn)
    general_info_df = pd.DataFrame(general_info)


    #Read the excel file with patient data and convert it into a pandas dataframe
    features_controls_info = pd.read_excel(route_features_controls_bcn)
    features_patients_info = pd.read_excel(route_features_patients_bcn)
    features_info_df = pd.concat([features_controls_info, features_patients_info], ignore_index=True)

    #Print the number of subjects to check we have the same number of every type 
    print("Num FA_subjects_BCN", len(fa_names_bcn_csv))
    print("Num Func_subjects_BCN", len(func_names_bcn_csv))
    print("Num GM_subjects_BCN", len(gm_names_bcn_csv))


    #Read the first matrix in csv to know the dimensions
    
    FA = pd.read_csv(route_patients_FA_bcn + fa_names_bcn_csv[0], index_col=None, header=None)# read the csv file
    fa_array = np.array(FA)#[:,1:] 
    Func = pd.read_csv(route_patients_Func_bcn + func_names_bcn_csv[0], index_col=None, header=None)
    func_array = np.array(Func)#[:,1:]
    GM = pd.read_csv(route_patients_GM_bcn + gm_names_bcn_csv[0], index_col=None, header=None)
    gm_array = np.array(GM)#[:,1:]

    print("FA shape", fa_array.shape)
    print("Func shape", func_array.shape)
    print("GM shape", gm_array.shape)

    #Create a numpy array where we can store all the csv in a matrix format
    data_list_bcn = list()

    #Iterate through every csv file, convert it into numpy array and add it into a general "list" with all other matrices
    for fa, func, gm in zip(fa_names_bcn_csv, func_names_bcn_csv, gm_names_bcn_csv):
        #Check if all the datapoint names are the same
        assert fa[:8] == func[:8] and fa[:8] == gm[:8], "The names are not sorted"
        #If the name of the file starts with "c" it means it's a control, if it starts with "F" it means it's a patient
        if fa[0] == "c" or fa[0] == "s":
            if fa[0] == "c":
                name = fa[:9]
            elif fa[0] == "s":
                name = fa[:8]

            fa_class = general_info_df[general_info_df['id'] == name]
            edds = general_info_df[general_info_df['id'] == name]
            features = features_info_df[features_info_df['ID'] == name]
            print("Features", features, "\n") 
            #Read the csv files
            FA = pd.read_csv(route_controls_FA_bcn + fa, index_col=None, header=None)# read the csv file 
            Func = pd.read_csv(route_controls_Func_bcn + func, index_col=None, header=None)
            GM = pd.read_csv(route_controls_GM_bcn + gm, index_col=None, header=None)

            #Transform the csv to np array and drop the index column. Add the resulting array in the tuple 
            
            datapoint = (np.array(FA), np.array(Func), np.array(GM), -1, edds, features)
            # add the resulting datapoints  list
            data_list_bcn.append(datapoint)
        
        if fa[0] == "F":
            name = fa[:7]
        
        else:
            name = fa[:8]
  
        fa_class = general_info_df[general_info_df['id'] == name]
        edds = general_info_df[general_info_df['id'] == name]
        features = features_info_df[features_info_df['ID'] == name]
        if len(edds) == 0:
            edds = -1 #If the edds value is not present, we will assign it a value of -1
        else:
            edds = edds["edss"].values[0]
        if len(fa_class) != 0:
            fa_class = fa_class["mstype"].values[0]

            #Read the csv files
            FA = pd.read_csv(route_patients_FA_bcn + fa, index_col=None, header=None)# read the csv file 
            Func = pd.read_csv(route_patients_Func_bcn + func, index_col=None, header=None)
            GM = pd.read_csv(route_patients_GM_bcn + gm, index_col=None, header=None)

            #Transform the csv to np array and drop the index column. Add the resulting array in the tuple 
            
            datapoint = (np.array(FA), np.array(Func), np.array(GM), fa_class, edds, features)
            # add the resulting datapoints  list
            data_list_bcn.append(datapoint)
        else:
            "The patient", name, "is not in the excel file"
            pass

    return data_list_bcn


def normalize_matrix(matrix):
    ''' This function normalizes a matrix by subtracting the minimum value and dividing by the maximum value minus the minimum value '''

    min_matrix = np.min(matrix)
    max_matrix = np.max(matrix)

    normalized_matrix = (matrix - min_matrix) / (max_matrix - min_matrix + 1e-6)

    return normalized_matrix

def znormalize_matrix(matrix):
    ''' This function z-normalizes a matrix by subtracting the mean and dividing by the standard deviation '''

    mean_matrix = np.mean(matrix)
    std_matrix = np.std(matrix)

    znormalized_matrix = (matrix - mean_matrix) / (std_matrix + 1e-6)

    return znormalized_matrix

def create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap):
    ''' This function creates a clean dataset combining all the data from the Barcelona and Naples datasets.'''
    
    datalist_bcn = extract_data_bcn(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_classes_bcn)
    datalist_nap = extract_data_nap(route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_nap)
    
    #Print the number of patients in the Barcelona dataset
    print("BCN", len(datalist_bcn))

    #Print the number of each type of patients the Barcelona dataset
    
    ms_types_bcn = [datapoint[3] for datapoint in datalist_bcn]
    print("BCN healty", ms_types_bcn.count(-1))
    print("RRMS BCN:",ms_types_bcn.count(0))
    print("SPMS BCN:",ms_types_bcn.count(1))
    print("PPMS BCN:",ms_types_bcn.count(2))
    print("MS types BCN", np.unique(ms_types_bcn))
    print("\n")
    #Print the number of patients in the Naples dataset
    print("NAP", len(datalist_nap))
    #Print the number of each type of patients in the Naples dataset
    ms_types_nap = [datapoint[3] for datapoint in datalist_nap]
    print("NAP healty", ms_types_nap.count(-1))
    print("RRMS NAP:",ms_types_nap.count(0))
    print("SPMS NAP:",ms_types_nap.count(1))
    print("PPMS NAP:",ms_types_nap.count(2))
    print("MS types NAP", np.unique(ms_types_bcn))
    print("\n")

    #Concatenate the two lists
    datalist = datalist_bcn + datalist_nap

    print("Total number of patients", len(datalist))

    return datalist



if __name__ == "__main__":
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)

    ms_types = [datapoint[3] for datapoint in datalist]
   
    print("Healty:",ms_types.count(-1))
    print("RRMS:",ms_types.count(0))
    print("SPMS:",ms_types.count(1))
    print("PPMS:",ms_types.count(2))
    print("MS types", np.unique(ms_types))















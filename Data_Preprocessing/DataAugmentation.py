import numpy as np  
import matplotlib.pyplot as plt
from Data_Preprocessing.data_cleaning import create_clean_dataset
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import random 



# BARCELONA DATA 
route_patients = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES"
route_patients_FA_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FA/"
route_patients_Func_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FUNC/"
route_patients_GM_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/GM_networks/"

route_controls_FA_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/CONTROLES/FA/"
route_controls_Func_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/CONTROLES/FUNC/"
route_controls_GM_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/CONTROLES/GM_networks/"
route_classes_bcn = "Dades/DADES_BCN/DADES_HCB/subject_clinical_data_20201001.xlsx"

#NAPLES DATA 
route_patients_FA_nap = "Dades/DADES_NAP/Naples/DTI_networks/"
route_patients_Func_nap = "Dades/DADES_NAP/Naples/rsfmri_networks/"
route_patients_GM_nap = "Dades/DADES_NAP/Naples/GM_networks/"
route_classes_nap = "Dades/DADES_NAP/Naples/naples2barcelona_multilayer.xlsx"



def check_std(data):
    """
    Check the standard deviation of the data.
    
    PARAMETERS:
    
    data: list of data points
    
    OUTPUT:
    
    standard deviation of the data
    """
    matrix_FA_list = []
    matrix_Func_list = []
    matrix_GM_list = []

    
    for i in range(len(data)):

        matrix_FA_list.append(data[i][0])
        matrix_Func_list.append(data[i][1])
        matrix_GM_list.append(data[i][2])

    
    max_FA = np.max(matrix_FA_list, axis=0)
    max_Func = np.max(matrix_Func_list, axis=0)
    max_GM = np.max(matrix_GM_list, axis=0)

    min_FA = np.min(matrix_FA_list, axis=0)
    min_Func = np.min(matrix_Func_list, axis=0)
    min_GM = np.min(matrix_GM_list, axis=0)

    print("Max FA",max_FA.shape, )
    matrix_FA_list_norm = []
    matrix_Func_list_norm = []
    matrix_GM_list_norm = []

    print("Max FA",max_FA.shape, "Min FA",min_FA.shape, "Max - Min:", (max_FA - min_FA))
    for fa, func, gm in zip(matrix_FA_list, matrix_Func_list, matrix_GM_list):
        matrix_FA_list_norm.append((fa - min_FA) / (max_FA - min_FA+ 1e-6) )
        
        matrix_Func_list_norm.append((func - min_Func) / (max_Func - min_Func+ 1e-6))
        matrix_GM_list_norm.append((gm - min_GM) / (max_GM - min_GM+ 1e-6))


    print("matrix_FA_list",len(matrix_FA_list))
    print("matrix_Func_list",len(matrix_Func_list))
    print("matrix_GM_list",len(matrix_GM_list))

    mean_matrix_FA = np.mean(matrix_FA_list_norm, axis=0)
    mean_matrix_Func = np.mean(matrix_Func_list_norm, axis=0)
    mean_matrix_GM = np.mean(matrix_GM_list_norm, axis=0)
    print("\n")
    print("Max FA mean",np.max(mean_matrix_FA), "Min FA mean",np.min(mean_matrix_FA))
    print("Max Func mean",np.max(mean_matrix_Func), "Min Func mean",np.min(mean_matrix_Func))
    print("Max GM mean",np.max(mean_matrix_GM), "Min GM mean",np.min(mean_matrix_GM))
    print("\n")


    std_matrix_FA = np.std(matrix_FA_list_norm, axis=0)
    std_matrix_Func = np.std(matrix_Func_list_norm, axis=0)
    std_matrix_GM = np.std(matrix_GM_list_norm, axis=0)

    
    return std_matrix_FA, std_matrix_Func, std_matrix_GM

def mix_single_graph(matrix1,matrix2, mixing_level, max_samples):

    new_matrix1 = matrix1.copy()
    new_matrix2 = matrix2.copy()
    matrices = [matrix1, matrix2]
    if mixing_level == 0:
        return matrix1, matrix2

    elif mixing_level == 1:
    
        new_matrix1[0:new_matrix1.shape[0]//2, 0:new_matrix1.shape[1]//2] = matrix1[0:new_matrix1.shape[0]//2, 0:new_matrix1.shape[1]//2]
        new_matrix1[new_matrix1.shape[0]//2:, 0:new_matrix1.shape[1]//2] = matrix2[new_matrix1.shape[0]//2:, 0:new_matrix1.shape[1]//2]
        new_matrix1[0:new_matrix1.shape[0]//2, new_matrix1.shape[1]//2:] = matrix1[0:new_matrix1.shape[0]//2, new_matrix1.shape[1]//2:]
        new_matrix1[new_matrix1.shape[0]//2:, new_matrix1.shape[1]//2:] = matrix2[new_matrix1.shape[0]//2:, new_matrix1.shape[1]//2:]

        new_matrix2[0:new_matrix2.shape[0]//2, 0:new_matrix2.shape[1]//2] = matrix2[0:new_matrix2.shape[0]//2, 0:new_matrix2.shape[1]//2]
        new_matrix2[new_matrix2.shape[0]//2:, 0:new_matrix2.shape[1]//2] = matrix1[new_matrix2.shape[0]//2:, 0:new_matrix2.shape[1]//2]
        new_matrix2[0:new_matrix2.shape[0]//2, new_matrix2.shape[1]//2:] = matrix2[0:new_matrix2.shape[0]//2, new_matrix2.shape[1]//2:]
        new_matrix2[new_matrix2.shape[0]//2:, new_matrix2.shape[1]//2:] = matrix1[new_matrix2.shape[0]//2:, new_matrix2.shape[1]//2:]

    elif mixing_level >= 2:
        step = matrix1.shape[0]//(2**mixing_level)
        for i in range(0, matrix1.shape[0], step):
            for j in range(0, matrix1.shape[1], step):
                new_matrix1[i:i+step, j:j+step] = matrices[np.random.randint(0,1)][i:i+step, j:j+step]
                new_matrix2[i:i+step, j:j+step] = matrices[np.random.randint(0,1)][i:i+step, j:j+step]



    return new_matrix1, new_matrix2
    

def datalist_mixing(datalist, mixing_levels, max_samples):
    """
    This function creates a new dataset by mixing the matrices of the original dataset. 
    The mixing will be done by dividing each matrix of same type in different regions of equal size, 
    and changing the regions between different matrices of the same type. 
    """

    new_datalist = []

    matrix_FA_list = []
    matrix_Func_list = []
    matrix_GM_list = []
    mstype_list = []
    edds_list = []  

    data_dict = {0:[], 1:[], 2:[], -1:[]}  
    base_num_datapoints = 0 #track the number of datapoints in the original dataset
    for datapoint in datalist:
        data_dict[datapoint[3]].append(datapoint)
        new_datalist.append(datapoint)
        base_num_datapoints += 1 

    for key in data_dict.keys():
        print("key",key, len(data_dict[key]))
        list_idx1 = []
        count_new_datapoints = base_num_datapoints
        for idx1,datapoint1 in enumerate(data_dict[key]):
            list_idx1.append(idx1)
            for idx2,datapoint2 in enumerate(data_dict[key]):
                if (idx2 not in list_idx1):
                    if (count_new_datapoints <= max_samples):
                        for level in mixing_levels:
                            if level != 0: 
                                new_fa_matrix1, new_fa_matrix2 = mix_single_graph(datapoint1[0],datapoint2[0],level, max_samples)
                                new_func_matrix1, new_func_matrix2 = mix_single_graph(datapoint1[1],datapoint2[1], level, max_samples)
                                new_gm_matrix1, new_gm_matrix2 = mix_single_graph(datapoint1[2],datapoint2[2], level, max_samples)

                                matrix_FA_list.append(new_fa_matrix1)
                                matrix_Func_list.append(new_func_matrix1)
                                matrix_GM_list.append(new_gm_matrix1)
                                mstype_list.append(datapoint1[3])
                                edds_list.append(datapoint1[4])

                                new_datalist.append((new_fa_matrix1, new_func_matrix1, new_gm_matrix1, datapoint1[3], datapoint1[4]))


                                matrix_FA_list.append(new_fa_matrix2)
                                matrix_Func_list.append(new_func_matrix2)
                                matrix_GM_list.append(new_gm_matrix2)
                                mstype_list.append(datapoint2[3])
                                edds_list.append(datapoint2[4])

                                new_datalist.append((new_fa_matrix2, new_func_matrix2, new_gm_matrix2, datapoint2[3], datapoint2[4]))

                                count_new_datapoints += 2

    return new_datalist


def add_noise(matrix, low = 0, high = 0.1):
    np.random.uniform(low, high, matrix.shape)
    matrix += np.random.uniform(low, high, matrix.shape)
    return matrix


def datalist_mixing_balanced(datalist, mixing_levels, max_samples):
    """
    This function creates a new dataset by mixing the matrices of the original dataset. 
    The mixing will be done by dividing each matrix of same type in different regions of equal size, 
    and changing the regions between different matrices of the same type. 
    """

    new_datalist = []

    matrix_FA_list = []
    matrix_Func_list = []
    matrix_GM_list = []
    mstype_list = []
    edds_list = []  
 
    data_dict = {0:[], 1:[], 2:[], -1:[]}  
    base_num_datapoints = 0 #track the number of datapoints in the original dataset
    count_num_datapoints_x_class = {0:0, 1:0, 2:0, -1:0} #track the number of datapoints in the original dataset by class
    for datapoint in datalist:
        data_dict[datapoint[3]].append(datapoint)
        new_datalist.append(datapoint)
        count_num_datapoints_x_class[datapoint[3]] += 1
        base_num_datapoints += 1 

    

    for key in data_dict.keys():
        print("key",key, len(data_dict[key]))
        list_idx1 = []
        count_new_datapoints = base_num_datapoints
        for idx1,datapoint1 in enumerate(data_dict[key]):
            list_idx1.append(idx1)
            for idx2,datapoint2 in enumerate(data_dict[key]):
                if (idx2 not in list_idx1):
                    if (count_num_datapoints_x_class[datapoint1[3]] <= max_samples):
                        for level in mixing_levels:
                            if level != 0: 
                                new_fa_matrix1, new_fa_matrix2 = mix_single_graph(datapoint1[0],datapoint2[0],level, max_samples)
                                new_func_matrix1, new_func_matrix2 = mix_single_graph(datapoint1[1],datapoint2[1], level, max_samples)
                                new_gm_matrix1, new_gm_matrix2 = mix_single_graph(datapoint1[2],datapoint2[2], level, max_samples)
                                
                                '''# Add noise to the new matrices
                                new_fa_matrix1 = add_noise(new_fa_matrix1)
                                new_fa_matrix2 = add_noise(new_fa_matrix2)
                                new_func_matrix1 = add_noise(new_func_matrix1)
                                new_func_matrix2 = add_noise(new_func_matrix2)
                                new_gm_matrix1 = add_noise(new_gm_matrix1)
                                new_gm_matrix2 = add_noise(new_gm_matrix2)'''
                            


                                matrix_FA_list.append(new_fa_matrix1)
                                matrix_Func_list.append(new_func_matrix1)
                                matrix_GM_list.append(new_gm_matrix1)
                                mstype_list.append(datapoint1[3])
                                edds_list.append(datapoint1[4])

                                new_datalist.append((new_fa_matrix1, new_func_matrix1, new_gm_matrix1, datapoint1[3], datapoint1[4], datapoint1[5],datapoint1[6]))


                                matrix_FA_list.append(new_fa_matrix2)
                                matrix_Func_list.append(new_func_matrix2)
                                matrix_GM_list.append(new_gm_matrix2)
                                mstype_list.append(datapoint2[3])
                                edds_list.append(datapoint2[4])

                                new_datalist.append((new_fa_matrix2, new_func_matrix2, new_gm_matrix2, datapoint2[3], datapoint2[4], datapoint2[5],datapoint2[6]))

                                count_new_datapoints += 2

                                count_num_datapoints_x_class[datapoint1[3]] += 2

    '''# Group datapoints by class (element at index 4)
    class_groups = defaultdict(list)
    for item in new_datalist:
        class_label = item[3]
        class_groups[class_label].append(item)
    # Print the number of items in each class
    for class_label, items in class_groups.items():
        print(f"Class {class_label}: {len(items)} items")
    
    # Find the minimum class size
    min_class_size = min(len(items) for items in class_groups.values())
    
    # Prune each class to the minimum size
    balanced_data = []
    for class_label, items in class_groups.items():
        balanced_data.extend(random.sample(items, min_class_size))'''

    return new_datalist
    

    
    


if __name__ == "__main__":
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    std_matrix_FA, std_matrix_Func, std_matrix_GM = check_std(datalist)
    print("std_matrix_FA",std_matrix_FA.shape)
    print("std_matrix_Func",std_matrix_Func.shape)
    print("std_matrix_GM",std_matrix_GM.shape)

    # Plot the standard deviation of the FA matrix
    plt.imshow(std_matrix_FA, cmap="hot")
    plt.title("Standard deviation FA")
    plt.colorbar()
    plt.show()
    print("\n")
    print("Average FA",np.mean(std_matrix_FA), "Max FA",np.max(std_matrix_FA), "Min FA",np.min(std_matrix_FA))
    print("\n")

    # Plot the standard deviation of the Func matrix
    plt.imshow(std_matrix_Func, cmap="hot")
    plt.title("Standard deviation Func")
    plt.colorbar()
    plt.show()
    print("\n")
    print("Average Func",np.mean(std_matrix_Func), "Max Func",np.max(std_matrix_Func), "Min Func",np.min(std_matrix_Func))
    print("\n")

    # Plot the standard deviation of the GM matrix
    plt.imshow(std_matrix_GM, cmap="hot")
    plt.title("Standard deviation GM")
    plt.colorbar()
    plt.show()
    print("\n")
    print("Average GM",np.mean(std_matrix_GM),"Max GM",np.max(std_matrix_GM), "Min GM",np.min(std_matrix_GM))
    print("\n")

    mixed_datalist = datalist_mixing_balanced(datalist, [1,2,3], 1000)
    print("mixed_datalist",len(mixed_datalist[0]))

    data_dict_final1 = {0:[], 1:[], 2:[], -1:[]}  
    for mixed_datapoint in mixed_datalist:
        data_dict_final1[mixed_datapoint[3]].append(mixed_datapoint)
        

    print("data_dict_final",[(key1, len(val)) for key1,val in data_dict_final1.items()])










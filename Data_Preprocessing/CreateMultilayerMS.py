import numpy as np 
from Data_Preprocessing.data_cleaning import create_clean_dataset



def create_multilayer_ms(datalist):
    

    multilayer_list = []    

    
    for i in range(len(datalist)):
        multilayer_matrix = np.zeros((2*datalist[i][0].shape[0], 2*datalist[i][0].shape[1]))
        multilayer_matrix[:76, :76] = datalist[i][0]
        multilayer_matrix[76:, :76] = datalist[i][1]
        multilayer_matrix[:76, 76:] = datalist[i][1]
        multilayer_matrix[76:, 76:] = datalist[i][2]
        multilayer_list.append((multilayer_matrix, datalist[i][3], datalist[i][4]))

    return multilayer_list
        
if __name__ == "__main__":

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


    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    multilayer_list = create_multilayer_ms(datalist)
    print("multilayer_list",len(multilayer_list))
    print("multilayer_list",multilayer_list[0][0].shape)

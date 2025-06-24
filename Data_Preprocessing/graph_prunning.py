from scipy.fft import dct, idct
import numpy as np
import sys
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG")
from Data_Preprocessing.data_cleaning import create_clean_dataset, normalize_matrix, znormalize_matrix
from Data_Preprocessing.CreateMultilayerMS import create_multilayer_ms
import networkx as nx
import matplotlib.pyplot as plt

def graph_prunning(datalist):
    pass

def dct2(s):
        return dct(dct(s.T, norm='ortho').T, norm='ortho')

def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def graph_prunning_DCT(datalist_multilayer, tau=0.1, STEP=1):
    p = 8
    M = p*p
    D = np.zeros((M, M))
    cnt = 0
    img_hat_list = []
    

    for i in range(p):
        for j in range(p):
            delta = np.zeros((p, p))
            delta[i, j] = 1
            D[cnt, :] = dct2(delta).reshape((-1))
            cnt += 1

    for idx, datapoint in enumerate(datalist_multilayer):
        img_hat = np.zeros_like(datalist_multilayer[0][0])
        weights = np.zeros_like(datalist_multilayer[0][0])
        print("datapoint", idx) 
        for i in range(0, datapoint[0][0].shape[0] - p + 1, STEP):
            for j in range(0, datapoint[0][0].shape[0] - p + 1, STEP):
                # extrach the patch with the top left corner at pixel (ii, jj)
                
                s = datapoint[0][i:i+p, j:j+p]
                

                # compute the representation w.r.t. the 2D DCT dictionary
                x = D.T @ s.reshape((-1, 1))
                x = x.reshape((p, p))
                
                # perform the hard thresholding (do not perform HT on the DC!)
                x_HT = x.reshape(-1)  # Reshape to 1D for thresholding
                x_HT = np.where(np.abs(x_HT) > tau, x_HT, 0)  # Thresholding
                
                #x_HT = x_HT.reshape(p, p)  # Reshape back to 2D


                # perform the reconstruction
                s_hat = D @ x_HT

                # compute the weights to be used for aggregating the reconstructed patch
                try: 
                    w = 1/(np.sum(np.abs(x)) + 1e-6)
                except:
                    print("W",w)
                # accumulate by summation the denoised patch into the denoised image using the computed weight
                # update img_hat
                
                    img_hat[i:i+p, j:j+p] += (w * s_hat).reshape((p,p))
                


                # accumulate by summation the weights of the current patch in the weight matrix
                # update weights

                weights[i:i+p, j:j+p] += w


        img_hat = img_hat / (weights +1e-6)
        img_hat_list.append((img_hat, datapoint[1], datapoint[2]))
    return img_hat_list


def graph_prunning_threshold(matrix, percentage):
    
    len_flat_datapoint = len(matrix.reshape(-1))
    threshold_value = np.sort(matrix.reshape(-1))[int(len_flat_datapoint*percentage)]
    prunned_matrix = np.where(matrix < threshold_value, 0, matrix)
    
    return prunned_matrix


def count_nonzeros_datalist(datalist):
    total_nonzeros = 0 

    for idx,datapoint in enumerate(datalist):
        print("datapoint num:s", idx)
        nonzero_num = np.count_nonzero(datapoint[0])
        total_nonzeros += nonzero_num
    
    return total_nonzeros

def normalize_and_threshold(datalist, threshold):
    normalized_datalist = []
    for idx,datapoint in enumerate(datalist):
        fa_matrix_norm = znormalize_matrix(datapoint[0])
        func_matrix_norm = znormalize_matrix(datapoint[1])
        gm_matrix_norm = znormalize_matrix(datapoint[2])

        fa_matrix_prune = graph_prunning_threshold(fa_matrix_norm, threshold)
        func_matrix_prune = graph_prunning_threshold(func_matrix_norm, threshold)
        gm_matrix_prune = graph_prunning_threshold(gm_matrix_norm, threshold)

        normalized_datalist.append((fa_matrix_prune, func_matrix_prune, gm_matrix_prune, datapoint[3], datapoint[4], datapoint[5], datapoint[6]))
    
    return normalized_datalist

if __name__ == "__main__":
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


    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)

    
    prunning_threshold = 0.7

    multilayer_list = create_multilayer_ms(datalist)
    
    datalist_normalized = normalize_and_threshold(datalist, prunning_threshold)
    
    multilayer_list_prune = create_multilayer_ms(datalist_normalized)

    #img_hat_list = graph_prunning_DCT(multilayer_list, tau=2, STEP=1)
    #prunned_datalist = graph_prunning_threshold(multilayer_list, 0.7)

    
    nonzero_multilayer = count_nonzeros_datalist(multilayer_list)
    #nonzero_img_hat = count_nonzeros_datalist(img_hat_list)
    nonzero_prunned = count_nonzeros_datalist(multilayer_list_prune)
    print("nonzero_multilayer", nonzero_multilayer)
    #print("nonzero_img_hat", nonzero_img_hat)
    print("nonzero_prunned", nonzero_prunned)

    # Assuming `graph` is your NetworkX graph
    # Convert the graph to an adjacency matrix
    adj_matrix_DTI = datalist[0][0]
    adj_matrix_fMRI = datalist[0][1]
    adj_matrix_GM = datalist[0][2]
    # Plot the adjacency matrix
    plt.figure(figsize=(8, 8))
    plt.title("Adjacency Matrix DTI")
    plt.imshow(adj_matrix_DTI, cmap='viridis', interpolation='none')
    plt.colorbar(label="Connection Strength")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title("Adjacency Matrix fMRI")
    plt.imshow(adj_matrix_fMRI, cmap='viridis', interpolation='none')
    plt.colorbar(label="Connection Strength")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title("Adjacency Matrix GM")
    plt.imshow(adj_matrix_GM, cmap='viridis', interpolation='none')
    plt.colorbar(label="Connection Strength")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.show()

    adj_matrix_norm_DTI = datalist_normalized[0][0]
    # Plot the adjacency matrix
    plt.figure(figsize=(8, 8))
    plt.title("Normalized Adjacency Matrix DTI")
    plt.imshow(adj_matrix_norm_DTI, cmap='viridis', interpolation='none')
    plt.colorbar(label="Connection Strength")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.show()

    adj_matrix_norm_fMRI = datalist_normalized[0][1]
    # Plot the adjacency matrix
    plt.figure(figsize=(8, 8))
    plt.title("Normalized Adjacency Matrix fMRI")
    plt.imshow(adj_matrix_norm_fMRI, cmap='viridis', interpolation='none')
    plt.colorbar(label="Connection Strength")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.show()

    adj_matrix_norm_GM = datalist_normalized[0][2]
    # Plot the adjacency matrix
    plt.figure(figsize=(8, 8))
    plt.title("Normalized Adjacency Matrix GM")
    plt.imshow(adj_matrix_norm_GM, cmap='viridis', interpolation='none')
    plt.colorbar(label="Connection Strength")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.show()
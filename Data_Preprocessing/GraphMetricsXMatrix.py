import os
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'
import networkx as nx
from data_cleaning import create_clean_dataset
from graph_prunning import graph_prunning_threshold, normalize_and_threshold
import numpy as np 
import time 
import pandas as pd
import pickle


import sys
sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Data_Preprocessing")


from data_cleaning import create_clean_dataset
from graph_prunning import normalize_and_threshold
from torch.utils.data import Dataset, DataLoader
from DataAugmentation import datalist_mixing, datalist_mixing_balanced
from CreateMultilayerMS import create_multilayer_ms

sys.path.append("/home/samper12/Escritorio/IA carrera/TFG/codi_TFG/Baseline_Models/SVM_models")
from Voting_SVM import generate_train_test_datalist



# BARCELONA DATA 
route_patients = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES"
route_patients_FA_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FA/"
route_patients_Func_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/FUNC/"
route_patients_GM_bcn = "Dades/DADES_BCN/DADES_HCB/MATRICES/PACIENTES/GM_networks/"
route_classes_bcn = "Dades/DADES_BCN/DADES_HCB/subject_clinical_data_20201001.xlsx"
route_volumes_nodes_controls_bcn = "Dades/DADES_BCN/DADES_HCB/VOLUM_nNODES_CONTROLS.xls"
route_volumes_nodes_patients_bcn = "Dades/DADES_BCN/DADES_HCB/VOLUM_nNODES_PATIENTS.xls"



#NAPLES DATA 
route_patients_FA_nap = "Dades/DADES_NAP/Naples/DTI_networks/"
route_patients_Func_nap = "Dades/DADES_NAP/Naples/rsfmri_networks/"
route_patients_GM_nap = "Dades/DADES_NAP/Naples/GM_networks/"
route_classes_nap = "Dades/DADES_NAP/Naples/naples2barcelona_multilayer.xlsx"
route_volumes_nodes_controls_nap = "Dades/DADES_NAP/Naples/NODES_NAPLES.csv"




datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)


# For a single node's local efficiency
def node_local_efficiency(G, node):
    """Calculate local efficiency for a single node"""
    neighbors = list(G.neighbors(node))
    
    if len(neighbors) < 2:
        return 0.0
    
    # Create subgraph of neighbors (excluding the node itself)
    subgraph = G.subgraph(neighbors)
    
    # Use the built-in global_efficiency function on this subgraph
    return nx.global_efficiency(subgraph)

def create_graph_list_x_matrix(datalist):
    """
    Create graphs from matrices.
    """
    graph_datalist = []
    
    for datapoint in datalist:
        
        G_Func = nx.Graph()
        G_FA = nx.Graph()
        G_GM = nx.Graph()   
        FA_matrix, Func_matrix, GM_matrix, mstype, edds, name, dataset_name = datapoint[0], datapoint[1], datapoint[2], datapoint[3], datapoint[4], datapoint[5],datapoint[6]
        #Add nodes to the graph
        for idx_node in range(FA_matrix.shape[0]):
            G_FA.add_node(str(idx_node))

        for idx_node in range(Func_matrix.shape[0]):
            G_Func.add_node(str(idx_node))

        for idx_node in range(GM_matrix.shape[0]):
            G_GM.add_node(str(idx_node))

        

        
        for i in range(FA_matrix.shape[0]):
            for j in range(FA_matrix.shape[1]):
                if FA_matrix[i,j] != 0 or i != j:
                    
                    node1 = str(i)
                    node2 = str(j)
                    G_FA.add_edge(node1, node2, weight=FA_matrix[i,j])

        for i in range(Func_matrix.shape[0]):
            for j in range(Func_matrix.shape[1]):
                if Func_matrix[i, j] != 0 or i != j:
                
                    node1 = str(i)
                    node2 = str(j)
                    G_Func.add_edge(node1, node2, weight=Func_matrix[i, j])

        
        for i in range(GM_matrix.shape[0]):
            for j in range(GM_matrix.shape[1]):
                if GM_matrix[i, j] != 0 or i != j:

                    node1 = str(i)
                    node2 = str(j)
                    G_GM.add_edge(node1, node2, weight=GM_matrix[i, j])


        # Append graph to list
        graph_datalist.append(((G_FA, G_Func, G_GM), mstype, edds, name, dataset_name))

    return graph_datalist

def compute_metrics_x_matrix(graphlist, features_dict, matrix_type = 'FA'):
    """
    Compute metrics for each graph in the datalist. 
    Each graph will have a new value in each node with the name of the specific metric and the computed value 

    PARAMETERS:
    graphlist: List of graphs to compute metrics for. Each graph is a tuple of the form (graph, mstype, edds, name).

    OUTPUT:
    new_graphlist: List of graphs with metrics appended. Each graph is a tuple of the form (graph, mstype, edds, name).

    """
    if matrix_type == 'FA':
        print("Computing metrics for FA graphs")
        matrix_num = 0
    elif matrix_type == 'Func':
        print("Computing metrics for Func graphs")
        matrix_num = 1
    elif matrix_type == 'GM':
        print("Computing metrics for GM graphs")
        matrix_num = 2
    # Create a new list to store the graphs with metrics
    new_graphlist = []
    for idx, datapoint in enumerate(graphlist):
        name = datapoint[3]
        # Initialize the features dictionary for each graph
        features_dict[name] = np.zeros((76,9))

    # Iterate over the graphlist and compute metrics for each graph
    for idx,datapoint in enumerate(graphlist):
        print(f"Computing metrics for graph {idx+1}")
        #Obtain the original graph to add the values
        graph = datapoint[0][matrix_num]
        name = datapoint[3]
        

        betweenness_centrality = nx.betweenness_centrality(graph)
        pagerank = nx.pagerank(graph, weight='weight')
        
        
        
        for node in graph.nodes():
            # Calculate local efficiency for this node
            local_eff = node_local_efficiency(graph, node)
            # Compute metrics for each node in the graph
            node_degree = nx.degree(graph, node)
            node_strength = nx.degree(graph, node, weight='weight')
            #shortest_path = nx.shortest_path(graph, source=node)
            triangles = nx.triangles(graph)[node]
            closeness_centrality = nx.closeness_centrality(graph, node)
            node_betweenness_centrality = betweenness_centrality[node]
            clustering = nx.clustering(graph)[node]
            
            # Append the metrics to the graph
            graph.nodes[node]['degree'] = node_degree
            graph.nodes[node]['strength'] = node_strength
            #graph.nodes[node]['shortest_path'] = shortest_path
            graph.nodes[node]['triangles'] = triangles
            graph.nodes[node]['closeness_centrality'] = closeness_centrality
            graph.nodes[node]['betweenness_centrality'] = node_betweenness_centrality
            graph.nodes[node]['clustering'] = clustering
            graph.nodes[node]['local_efficiency'] = local_eff
            graph.nodes[node]['pagerank'] = pagerank[node]

            # Append all the metrcs to the features dictionary and a 0 at the end to later save the volume of every node
            features_node = np.array([node_degree, node_strength, triangles, closeness_centrality, node_betweenness_centrality, clustering, local_eff, pagerank[node],0])
            
            features_dict[name][int(node), :] = features_node
            
        # Append the graph to the new graph list, along with the mstype, edds, and name
        new_graphlist.append((graph, datapoint[1],datapoint[2], datapoint[3]))

    return new_graphlist, features_dict

def append_graph_volumes(graph_volumes_controls_bcn_url,graph_volumes_patients_bcn_url, graph_volumes_nap_url, graphlist, features_dict):
    """
    Append graph volumes to the graph list. 
    Each node of the graph will be filled with a new value with the name 'volumes'. 
    The value will be the normalized volume of the node in the graph. 
    The volume is taken from the graph volumes file and normalized to 0-1 range for every row of values.
    Each row of values contains all the parts of the brain.

    PARAMETERS:
    graph_volumes_controls_bcn_url: URL to the controls volumes file
    graph_volumes_patients_bcn_url: URL to the patients volumes file
    graph_volumes_nap_url: URL to the Naples volumes file
    graphlist: List of graphs to which the volumes will be appended 

    OUTPUT: 
    graphlist: List of graphs with volumes appended 

    As the our graph has 76x3 nodes as we have 3 different types of data in the same graph.
    But our data only have 76 nodes, so we will add the same data 3 times. 
    On the corresponding nodes of each different type of graph.  

    """

    # Read the Excel and CSV files
    graph_volumes_controls_bcn = pd.read_excel(graph_volumes_controls_bcn_url)
    graph_volumes_patients_bcn = pd.read_excel(graph_volumes_patients_bcn_url)
    graph_volumes_naples = pd.read_csv(graph_volumes_nap_url, delimiter = " ")
    graph_volumes_naples = pd.DataFrame(graph_volumes_naples)

    # Merge the two datasets into a single DataFrame
    graph_volumes = pd.concat([graph_volumes_controls_bcn, graph_volumes_patients_bcn, graph_volumes_naples], ignore_index=True)
    
    # Make a copy to avoid modifying the original dataframe
    graph_volumes = graph_volumes.copy()

    # First ensure we have ID column separated from data columns
    id_column = graph_volumes['ID'].copy()
    data_columns = graph_volumes.drop(columns=['ID'])

    # Normalize data columns row by row (each patient's volumes normalized to 0-1 range)
    normalized_data = data_columns.div(data_columns.max(axis=1), axis=0).fillna(0)

    # Re-combine ID column with normalized data
    graph_volumes = pd.concat([id_column, normalized_data], axis=1)

    # Remove the unused column 'Unnamed: 77' if it exists
    if 'Unnamed: 77' in graph_volumes.columns:
        graph_volumes = graph_volumes.drop(columns=['Unnamed: 77'])

    # Iterate over the graphlist and append the volumes to each graph
    for idx, graph_data in enumerate(graphlist):
        graph = graph_data[0]
        mstype = graph_data[1]
        edds = graph_data[2]
        name = graph_data[3]

        
        real_name = name
        # Some names lack the letter 'r' at the beginning, so we add it to have a perfect string match
        if "r" + name in graph_volumes['ID'].values:
            name = "r" + name
        
        # Check if the name exists in the graph_volumes DataFrame
        if name in graph_volumes['ID'].values:
            # Extract the row of volume values corresponding to the current name
            volume_values = graph_volumes.loc[graph_volumes["ID"] == name].reset_index(drop=True)
            
            # Check ifopen_features_dict the volume_values DataFrame is not empty
            if not volume_values.empty:
                volume_values = volume_values.iloc[0][1:]  # Take the first matching row and remove the ID column

            #Handle exceptions
            else:
                print(f"Warning: No volume data found for ID {name}")
                volume_values = pd.Series(0, index=graph_volumes.columns)  # Create empty series
        #Handle the case where the ID is not found
        else:
            print(f"Warning: ID {name} not found in volume data")
            volume_values = pd.Series(0, index=graph_volumes.columns)  # Create empty series
        
        # Assign the volume values to the corresponding nodes in the graph
        for idx_node,node in enumerate(graph.nodes()):
            real_idx_node = idx_node
            new_idx_node = idx_node   
            # As we only have 76 values but 76x3 nodes we assign same values multiple times 
            if 76 <= idx_node < 152:
                new_idx_node = idx_node - 76

            elif 152 <= idx_node:
                new_idx_node = idx_node - 152
         
            graph.nodes[node]['volumes'] = volume_values.values[new_idx_node]
            new_features_vol = features_dict[real_name][real_idx_node]
            new_features_vol[-1] = volume_values.values[new_idx_node]
            features_dict[real_name][real_idx_node] = new_features_vol

        graphlist[idx] = (graph, mstype, edds, name)

    print("features_dict", features_dict)

    return graphlist, features_dict

def save_graphlist(graphlist, filepath):
    """
    Save the graphlist to a file. 
    Each graph will be saved in a separate file with the name of the patient. 
    The graph will be saved in GML format. 

    PARAMETERS:
    graphlist: List of graphs to save. Each graph is a tuple of the form (graph, mstype, edds, name).

    OUTPUT:
    None
    """
    # Create directory if it doesn't exist
    
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for datapoint in graphlist:
        graph = datapoint[0]
        name = datapoint[3]
        mapping = {n: f"n{n}" for n in graph.nodes()}
        string_graph = nx.relabel_nodes(graph, mapping)
        # Save the graph in pickle format
        nx.write_gml(string_graph, f"{filepath}/{name}.gpickle")

        print(f"Graph {name} saved to {filepath}/{name}.gpickle")

def open_graphs(graphlist_path, datalist):
    """
    Open the graphs from a file. 
    Each graph will be opened from a separate file with the name of the patient. 
    The graph will be opened in GML format. 

    PARAMETERS:
    graphlist_path: Path to the folder containing the graphs.
    datalist: List of graphs to open. Each graph is a tuple of the form (graph, mstype, edds, name).

    OUTPUT:
    None
    """
    graphlist = []
    for idx,datapoint in enumerate(datalist):
        print("Idx", idx)
        graph = datapoint[0]
        name = datapoint[5]
        
        # Open the graph in GML format
        graph = nx.read_gml(f"{graphlist_path}/{name}.gpickle")
        graphlist.append((graph, datapoint[3],datapoint[4], name))

    return graphlist



def save_features(features_dict, filepath):
    """
    Save the features dictionary to a file. 
    Each graph will be saved in a separate file with the name of the patient. 
    The graph will be saved in GML format. 

    PARAMETERS:
    features_dict: Dictionary of features to save. Each graph is a tuple of the form (graph, mstype, edds, name).

    OUTPUT:
    None
    """
    
        
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    

    for key, value in features_dict.items():
        with open(filepath +"/features_dict.pkl", 'wb') as f:
            pickle.dump(features_dict, f)
            print(f"Dictionary saved to {filepath}")


def open_features_dict( filepath):
    """
    Open the features dictionary from a file. 
    Each graph will be opened from a separate file with the name of the patient. 
    The graph will be opened in GML format. 

    PARAMETERS:
    features_dict_path: Path to the folder containing the features.

    OUTPUT:
    None
    """
    
    features_dict = {}
    
    with open(filepath+"/features_dict.pkl", 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

if __name__ == "__main__":
    
    datalist = create_clean_dataset(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_bcn, route_classes_nap)
    
    datalist_thresholded = normalize_and_threshold(datalist, 0.7)
    datalist_train, datalist_test = generate_train_test_datalist(datalist_thresholded, train_ratio = 0.7)
    datalist_augmented_train = datalist_mixing_balanced(datalist_train, [0,1,2,3], 110)
    
    print("datalist threshold done\n")
    print("datalist augmented", len(datalist_augmented_train))
    print("datalist test", len(datalist_test))
    graphlist_train= create_graph_list_x_matrix(datalist_augmented_train)
    graphlist_test= create_graph_list_x_matrix(datalist_test)

    # Initialize the features dictionary for each graph
    features_dict_train_FA = {}
    features_dict_train_Func = {}
    features_dict_train_GM = {}
    

    # Initialize the features dictionary for each graph
    features_dict_test_FA = {}
    features_dict_test_Func = {}
    features_dict_test_GM = {}


    # Define the train features for the FA graphs 
    graphlist_train_FA, features_dict_train_FA = compute_metrics_x_matrix(graphlist_train, features_dict_train_FA, matrix_type='FA')
    graphlist_train_FA, features_dict_train_FA = append_graph_volumes(route_volumes_nodes_controls_bcn, route_volumes_nodes_patients_bcn, route_volumes_nodes_controls_nap, graphlist_train_FA, features_dict_train_FA)
    
    # Define the test features for the FA graphs 
    graphlist_test_FA, features_dict_test_FA = compute_metrics_x_matrix(graphlist_test, features_dict_test_FA, matrix_type='FA')
    graphlist_test_FA, features_dict_test_FA = append_graph_volumes(route_volumes_nodes_controls_bcn, route_volumes_nodes_patients_bcn, route_volumes_nodes_controls_nap, graphlist_test_FA, features_dict_test_FA)
    
    print("Graph creation FA")
    print("Features dict train", len(features_dict_train_FA))
    print("Features dict test", len(features_dict_test_FA))
    print("Graph creation finished")

    save_graphlist(graphlist_train_FA, "graphs_dict_train_FA")
    save_features(features_dict_train_FA, "featues_dict_train_FA")
    print("Graphlist train saved")

    save_graphlist(graphlist_test_FA, "graphs_dict_test_FA")
    save_features(features_dict_test_FA, "featues_dict_test_FA")
    print("Graphlist test saved\n")


    # Define the train features for the Func graphs
    graphlist_train_Func, features_dict_train_Func = compute_metrics_x_matrix(graphlist_train, features_dict_train_Func, matrix_type='Func')
    graphlist_train_Func, features_dict_train_Func = append_graph_volumes(route_volumes_nodes_controls_bcn, route_volumes_nodes_patients_bcn, route_volumes_nodes_controls_nap, graphlist_train_Func, features_dict_train_Func)

    # Define the test features for the Func graphs
    graphlist_test_Func, features_dict_test_Func = compute_metrics_x_matrix(graphlist_test, features_dict_test_Func, matrix_type='Func')
    graphlist_test_Func, features_dict_test_Func = append_graph_volumes(route_volumes_nodes_controls_bcn, route_volumes_nodes_patients_bcn, route_volumes_nodes_controls_nap, graphlist_test_Func, features_dict_test_Func)

    print("Graph creation Func")
    print("Features dict train", len(features_dict_train_Func))
    print("Features dict test", len(features_dict_test_Func))
    print("Graph creation finished")

    save_graphlist(graphlist_train_Func, "graphs_dict_train_Func")
    save_features(features_dict_train_Func, "featues_dict_train_Func")
    print("Graphlist train saved")

    save_graphlist(graphlist_test_Func, "graphs_dict_test_Func")
    save_features(features_dict_test_Func, "featues_dict_test_Func")
    print("Graphlist test saved\n")


    # Define the features for the GM graphs
    graphlist_train_GM, features_dict_train_GM = compute_metrics_x_matrix(graphlist_train, features_dict_train_GM, matrix_type='GM')
    graphlist_train_GM, features_dict_train_GM = append_graph_volumes(route_volumes_nodes_controls_bcn, route_volumes_nodes_patients_bcn, route_volumes_nodes_controls_nap, graphlist_train_GM, features_dict_train_GM)

    # Define the features for the GM graphs
    graphlist_test_GM, features_dict_test_GM = compute_metrics_x_matrix(graphlist_test, features_dict_test_GM, matrix_type='GM')
    graphlist_test_GM, features_dict_test_GM = append_graph_volumes(route_volumes_nodes_controls_bcn, route_volumes_nodes_patients_bcn, route_volumes_nodes_controls_nap, graphlist_test_GM, features_dict_test_GM)
    
    print("Graph creation GM")
    print("Features dict train", len(features_dict_train_GM))
    print("Features dict test", len(features_dict_test_GM))
    print("Graph creation finished")

    save_graphlist(graphlist_train_GM, "graphs_dict_train_GM")
    save_features(features_dict_train_GM, "featues_dict_train_GM")
    print("Graphlist train saved")

    save_graphlist(graphlist_test_GM, "graphs_dict_test_GM")
    save_features(features_dict_test_GM, "featues_dict_test_GM")
    print("Graphlist test saved")

    #graphlist = open_graphs("computed_graphs", datalist)


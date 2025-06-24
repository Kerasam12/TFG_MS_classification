import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from data_cleaning import extract_data_bcn, extract_data_nap
from graph_prunning import normalize_and_threshold

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

def t_test(dataset1, dataset2, data_type='DTI', normalize = False):
    """
    Perform a t-test between two datasets, to see if they come from the same distribution.

    Parameters:
    dataset1 (list or array-like): First dataset.
    dataset2 (list or array-like): Second dataset.

    Returns:
    tuple: t-statistic and p-value of the t-test.
    """
    if data_type == 'DTI':
        print("Performing t-test for DTI data...")
        data_idx = 0  # Assuming DTI data is at index 0 in the tuple
    elif data_type == 'fMRI':
        print("Performing t-test for fMRI data...")
        data_idx = 1
    elif data_type == 'GM':
        print("Performing t-test for GM data...")
        data_idx = 2

    if normalize:
        print("Normalizing datasets...")
        dataset1 = normalize_and_threshold(dataset1, threshold=0.7)
        dataset2 = normalize_and_threshold(dataset2, threshold=0.7)


    sum_dataset1_DTI = []
    for datapoint in dataset1:
        sum_DTI = np.sum(datapoint[data_idx])
        sum_dataset1_DTI.append(sum_DTI)
    
    sum_dataset2_DTI = []
    for datapoint in dataset2:
        sum_DTI = np.sum(datapoint[data_idx])
        sum_dataset2_DTI.append(sum_DTI)


    t_statistic, p_value = stats.ttest_ind(sum_dataset1_DTI, sum_dataset2_DTI)

    # Your t-test
    t_statistic, p_value = stats.ttest_ind(sum_dataset1_DTI, sum_dataset2_DTI)

    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    # Plot the distributions
    plt.figure(figsize=(12, 8))

    # Subplot 1: Histograms
    plt.subplot(2, 2, 1)
    plt.hist(sum_dataset1_DTI, alpha=0.7, label='Dataset 1 (BCN)', bins=20, color='blue')
    plt.hist(sum_dataset2_DTI, alpha=0.7, label='Dataset 2 (NAP)', bins=20, color='red')
    plt.xlabel(f'{data_type} Sum Values')
    plt.ylabel('Frequency')
    plt.title('Distribution Comparison - Histograms')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Box plots
    plt.subplot(2, 2, 2)
    data_to_plot = [sum_dataset1_DTI, sum_dataset2_DTI]
    labels = ['Dataset 1 (BCN)', 'Dataset 2 (NAP)']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel(f'{data_type} Sum Values')
    plt.title('Distribution Comparison - Box Plots')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Violin plots
    plt.subplot(2, 2, 3)
    positions = [1, 2]
    plt.violinplot([sum_dataset1_DTI, sum_dataset2_DTI], positions=positions)
    plt.xticks(positions, labels)
    plt.ylabel(f'{data_type} Sum Values')
    plt.title('Distribution Comparison - Violin Plots')
    plt.grid(True, alpha=0.3)

    # Subplot 4: Statistical summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Statistical Test Results:

    T-statistic: {t_statistic:.4f}
    P-value: {p_value:.4f}
    Significance level: α = 0.05
    Result: {'Significant' if p_value < 0.05 else 'Not significant'}

    Dataset 1 (BCN):
    Mean: {np.mean(sum_dataset1_DTI):.4f}
    Std: {np.std(sum_dataset1_DTI):.4f}
    N: {len(sum_dataset1_DTI)}

    Dataset 2 (NAP):
    Mean: {np.mean(sum_dataset2_DTI):.4f}
    Std: {np.std(sum_dataset2_DTI):.4f}
    N: {len(sum_dataset2_DTI)}

    Mean Difference: {np.mean(sum_dataset1_DTI) - np.mean(sum_dataset2_DTI):.4f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.suptitle(f'DTI Sum Comparison: BCN vs NAP (p = {p_value:.4f})', y=1.02)
    plt.show()

    # Additional detailed plot with seaborn
    plt.figure(figsize=(10, 6))

    # Combine data for seaborn
    combined_data = np.concatenate([sum_dataset1_DTI, sum_dataset2_DTI])
    group_labels = ['BCN'] * len(sum_dataset1_DTI) + ['NAP'] * len(sum_dataset2_DTI)

    # Create DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame({
        data_type+'_Sum': combined_data,
        'Dataset': group_labels
    })

    # Seaborn distribution plot
    custom_colors = ['#4472C4', '#FF0000']  # Blue and intense red
    sns.histplot(data=df, x=data_type+'_Sum', hue='Dataset', alpha=0.6, kde=True, palette=custom_colors)

    plt.title(f'{data_type} Sum Distribution by Dataset (t = {t_statistic:.3f}, p = {p_value:.4f})')
    plt.xlabel(f'{data_type} Sum Values')
    plt.ylabel('Density')

    # Add vertical lines for means
    mean1 = np.mean(sum_dataset1_DTI)
    mean2 = np.mean(sum_dataset2_DTI)
    plt.axvline(mean1, color='blue', linestyle='--', alpha=0.8, label=f'dataset1 Mean: {mean1:.3f}')
    plt.axvline(mean2, color='red', linestyle='--', alpha=0.8, label=f'dataset2 Mean: {mean2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return t_statistic, p_value


if __name__ == "__main__":
    dataset1 = extract_data_bcn(route_patients_FA_bcn, route_patients_Func_bcn, route_patients_GM_bcn, route_classes_bcn)
    dataset2 = extract_data_nap(route_patients_FA_nap, route_patients_Func_nap, route_patients_GM_nap, route_classes_nap)
    

    # Perform t-test
    t_statistic, p_value = t_test(dataset1, dataset2, data_type='GM', normalize=True)
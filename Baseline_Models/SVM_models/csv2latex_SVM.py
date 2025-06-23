import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# List of kernel types
kernels = ['rbf', 'linear']

for kernel in kernels:
    # Load CSV file
    filename = f"svm_full_results_{kernel}.csv"
    df = pd.read_csv(filename)
    
    # Create directory for outputs
    os.makedirs("visualization_output", exist_ok=True)
    
    # Function to highlight max values
    def highlight_max(value, max_value):
        return f"\\textbf{{{value:.4f}}}" if abs(value - max_value) < 1e-6 else f"{value:.4f}"

    #----------------------------------------------
    # 1. Main Results Table (keep it focused)
    #----------------------------------------------
    main_cols = ["Threshold", "Mixing Level", "Mean Accuracy", "Mean Precision", 
                "Mean Recall", "Mean F1 Score"]
    main_df = df[main_cols].copy()
    
    # Apply bold formatting for max values
    for col in ["Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1 Score"]:
        max_value = main_df[col].max()
        main_df[col] = main_df[col].apply(lambda x: highlight_max(x, max_value))
    
    # Generate LaTeX table
    latex_table = main_df.to_latex(index=False, column_format="|c|c|c|c|c|c|", escape=False)
    
    # Save as .tex file
    tex_filename = f"visualization_output/svm_main_results_{kernel}.tex"
    with open(tex_filename, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main Results for SVM with " + kernel.upper() + " kernel}\n")
        f.write("\\label{tab:svm_main_" + kernel + "}\n")
        f.write(latex_table)
        f.write("\\end{table}\n")
    
    #----------------------------------------------
    # 2. Per-Class Accuracy Table
    #----------------------------------------------
    accuracy_cols = ["Threshold", "Mixing Level", 
                     "No MS Accuracy", "RRMS Accuracy", "SPMS Accuracy", "PPMS Accuracy"]
    if all(col in df.columns for col in accuracy_cols):
        acc_df = df[accuracy_cols].copy()
        
        # Apply bold formatting for max accuracy per class
        for col in ["No MS Accuracy", "RRMS Accuracy", "SPMS Accuracy", "PPMS Accuracy"]:
            max_value = acc_df[col].max()
            acc_df[col] = acc_df[col].apply(lambda x: highlight_max(x, max_value))
        
        # Generate LaTeX table
        latex_table = acc_df.to_latex(index=False, column_format="|c|c|c|c|c|c|", escape=False)
        
        # Save as .tex file
        tex_filename = f"visualization_output/svm_class_accuracy_{kernel}.tex"
        with open(tex_filename, "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Per-Class Accuracy for SVM with " + kernel.upper() + " kernel}\n")
            f.write("\\label{tab:svm_accuracy_" + kernel + "}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
    
    #----------------------------------------------
    # 3. Per-Class Precision and Recall Tables
    #----------------------------------------------
    precision_cols = ["Threshold", "Mixing Level", 
                      "No MS Precision", "RRMS Precision", "SPMS Precision", "PPMS Precision"]
    recall_cols = ["Threshold", "Mixing Level", 
                   "No MS Recall", "RRMS Recall", "SPMS Recall", "PPMS Recall"]
    
    if all(col in df.columns for col in precision_cols) and all(col in df.columns for col in recall_cols):
        # Precision table
        prec_df = df[precision_cols].copy()
        
        for col in ["No MS Precision", "RRMS Precision", "SPMS Precision", "PPMS Precision"]:
            max_value = prec_df[col].max()
            prec_df[col] = prec_df[col].apply(lambda x: highlight_max(x, max_value))
        
        latex_table = prec_df.to_latex(index=False, column_format="|c|c|c|c|c|c|", escape=False)
        
        tex_filename = f"visualization_output/svm_class_precision_{kernel}.tex"
        with open(tex_filename, "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Per-Class Precision for SVM with " + kernel.upper() + " kernel}\n")
            f.write("\\label{tab:svm_precision_" + kernel + "}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
        
        # Recall table
        recall_df = df[recall_cols].copy()
        
        for col in ["No MS Recall", "RRMS Recall", "SPMS Recall", "PPMS Recall"]:
            max_value = recall_df[col].max()
            recall_df[col] = recall_df[col].apply(lambda x: highlight_max(x, max_value))
        
        latex_table = recall_df.to_latex(index=False, column_format="|c|c|c|c|c|c|", escape=False)
        
        tex_filename = f"visualization_output/svm_class_recall_{kernel}.tex"
        with open(tex_filename, "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Per-Class Recall for SVM with " + kernel.upper() + " kernel}\n")
            f.write("\\label{tab:svm_recall_" + kernel + "}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
    
    #----------------------------------------------
    # 4. Create Heatmaps for Visual Comparison
    #----------------------------------------------
    # Pivot table for heatmap: rows=threshold, cols=mixing level, values=accuracy
    if 'Threshold' in df.columns and 'Mixing Level' in df.columns and 'Mean Accuracy' in df.columns:
        heatmap_data = df.pivot_table(index='Threshold', columns='Mixing Level', 
                                    values='Mean Accuracy', aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'Mean Accuracy'})
        plt.title(f'SVM with {kernel.upper()} kernel: Accuracy by Threshold and Mixing Level')
        plt.tight_layout()
        plt.savefig(f"visualization_output/svm_heatmap_{kernel}.png", dpi=300)
        plt.close()
        
        # Generate TikZ code for LaTeX
        tikz_filename = f"visualization_output/svm_heatmap_{kernel}.tex"
        with open(tikz_filename, "w") as f:
            f.write("\\begin{figure}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\includegraphics[width=0.8\\textwidth]{" + f"visualization_output/svm_heatmap_{kernel}.png" + "}\n")
            f.write("\\caption{Heatmap of Mean Accuracy for SVM with " + kernel.upper() + " kernel}\n")
            f.write("\\label{fig:svm_heatmap_" + kernel + "}\n")
            f.write("\\end{figure}\n")
    
    #----------------------------------------------
    # 5. Create Class Performance Comparison Chart
    #----------------------------------------------
    # Identify best configuration (highest mean accuracy)
    if 'Mean Accuracy' in df.columns:
        best_config_idx = df['Mean Accuracy'].idxmax()
        best_config = df.iloc[best_config_idx]
        
        # Extract per-class metrics for the best configuration
        class_names = ['No MS', 'RRMS', 'SPMS', 'PPMS']
        
        # Check if we have all the required columns
        accuracy_cols = [f"{cls} Accuracy" for cls in class_names]
        precision_cols = [f"{cls} Precision" for cls in class_names]
        recall_cols = [f"{cls} Recall" for cls in class_names]
        
        if all(col in df.columns for col in accuracy_cols + precision_cols + recall_cols):
            # Extract data
            accuracies = [best_config[f"{cls} Accuracy"] for cls in class_names]
            precisions = [best_config[f"{cls} Precision"] for cls in class_names]
            recalls = [best_config[f"{cls} Recall"] for cls in class_names]
            
            # Plot bar chart
            x = np.arange(len(class_names))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width, accuracies, width, label='Accuracy')
            ax.bar(x, precisions, width, label='Precision')
            ax.bar(x + width, recalls, width, label='Recall')
            
            ax.set_ylabel('Score')
            ax.set_title(f'Best Model Performance by Class (SVM {kernel.upper()})')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f"visualization_output/svm_class_comparison_{kernel}.png", dpi=300)
            plt.close()
            
            # Generate LaTeX figure
            fig_filename = f"visualization_output/svm_class_comparison_{kernel}.tex"
            with open(fig_filename, "w") as f:
                f.write("\\begin{figure}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\includegraphics[width=0.9\\textwidth]{" + f"visualization_output/svm_class_comparison_{kernel}.png" + "}\n")
                f.write(f"\\caption{{Performance by Class for Best SVM Model with {kernel.upper()} kernel (Threshold: {best_config['Threshold']}, Mixing Level: {best_config['Mixing Level']})}}\n")
                f.write("\\label{fig:svm_class_comp_" + kernel + "}\n")
                f.write("\\end{figure}\n")
    
    print(f"Completed visualization for {kernel} kernel - check the 'visualization_output' directory")

print("All visualizations created successfully!")
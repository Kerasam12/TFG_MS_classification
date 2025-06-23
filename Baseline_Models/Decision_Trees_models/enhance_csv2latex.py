import pandas as pd
import numpy as np
import os

# List of kernel types
kernels = ['rbf', 'linear']

# Ensure output directory exists
os.makedirs("enhanced_tables", exist_ok=True)

for kernel in kernels:
    # Load CSV file
    filename = f"svm_full_results_{kernel}.csv"
    df = pd.read_csv(filename)
    
    # Function to highlight max values with color
    def highlight_max(value, max_value, column_type='accuracy'):
        # Different colors for different metric types
        colors = {
            'accuracy': 'blue',
            'precision': 'green',
            'recall': 'orange',
            'f1': 'purple'
        }
        color = colors.get(column_type, 'blue')
        
        if abs(value - max_value) < 1e-6:
            return f"\\cellcolor{{light{color}}}\\textbf{{{value:.4f}}}"
        else:
            return f"{value:.4f}"
    
    # Best configuration table (simplified version with best results only)
    best_idx = df['Mean Accuracy'].idxmax()
    best_config = df.iloc[best_idx]
    
    # Create a nicely formatted top results table
    best_df = pd.DataFrame({
        'Metric': ['Best Configuration', 'Threshold', 'Mixing Level', 
                  'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'],
        'Value': ['', best_config['Threshold'], best_config['Mixing Level'],
                 f"{best_config['Mean Accuracy']:.4f}", 
                 f"{best_config['Mean Precision']:.4f}",
                 f"{best_config['Mean Recall']:.4f}", 
                 f"{best_config['Mean F1 Score']:.4f}"]
    })
    
    # Generate enhanced LaTeX table with color coding and better formatting
    tex_filename = f"enhanced_tables/svm_best_results_{kernel}.tex"
    with open(tex_filename, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Best SVM Results with " + kernel.upper() + " Kernel}\n")
        f.write("\\label{tab:svm_best_" + kernel + "}\n")
        f.write("\\begin{tabular}{|l|c|}\n")
        f.write("\\hline\n")
        f.write("\\rowcolor{gray!20} \\textbf{Metric} & \\textbf{Value} \\\\ \\hline\n")
        f.write("\\rowcolor{gray!10} \\multicolumn{2}{|c|}{\\textbf{Best Configuration}} \\\\ \\hline\n")
        f.write(f"Threshold & {best_config['Threshold']} \\\\ \\hline\n")
        f.write(f"Mixing Level & {best_config['Mixing Level']} \\\\ \\hline\n")
        f.write("\\rowcolor{gray!10} \\multicolumn{2}{|c|}{\\textbf{Performance Metrics}} \\\\ \\hline\n")
        f.write(f"Mean Accuracy & \\textcolor{{blue}}{{\\textbf{{{best_config['Mean Accuracy']:.4f}}}}} \\\\ \\hline\n")
        f.write(f"Mean Precision & \\textcolor{{green}}{{\\textbf{{{best_config['Mean Precision']:.4f}}}}} \\\\ \\hline\n")
        f.write(f"Mean Recall & \\textcolor{{orange}}{{\\textbf{{{best_config['Mean Recall']:.4f}}}}} \\\\ \\hline\n")
        f.write(f"Mean F1 Score & \\textcolor{{purple}}{{\\textbf{{{best_config['Mean F1 Score']:.4f}}}}} \\\\ \\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Create per-class comparison table for the best configuration
    class_names = ['No MS', 'RRMS', 'SPMS', 'PPMS']
    
    # Check if we have all the required columns
    accuracy_cols = [f"{cls} Accuracy" for cls in class_names]
    precision_cols = [f"{cls} Precision" for cls in class_names]
    recall_cols = [f"{cls} Recall" for cls in class_names]
    
    if all(col in df.columns for col in accuracy_cols + precision_cols + recall_cols):
        # Extract data for best configuration
        class_metrics = {
            'Class': class_names,
            'Accuracy': [best_config[f"{cls} Accuracy"] for cls in class_names],
            'Precision': [best_config[f"{cls} Precision"] for cls in class_names],
            'Recall': [best_config[f"{cls} Recall"] for cls in class_names]
        }
        
        # Find best performing class for each metric
        best_acc_class = np.argmax(class_metrics['Accuracy'])
        best_prec_class = np.argmax(class_metrics['Precision'])
        best_recall_class = np.argmax(class_metrics['Recall'])
        
        # Generate LaTeX table with color highlighting
        tex_filename = f"enhanced_tables/svm_class_performance_{kernel}.tex"
        with open(tex_filename, "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Per-Class Performance for Best SVM Model with " + kernel.upper() + " Kernel}\n")
            f.write("\\label{tab:svm_class_perf_" + kernel + "}\n")
            
            # Add color definitions for the table
            f.write("\\definecolor{lightblue}{rgb}{0.8,0.9,1.0}\n")
            f.write("\\definecolor{lightgreen}{rgb}{0.8,1.0,0.8}\n")
            f.write("\\definecolor{lightorange}{rgb}{1.0,0.9,0.8}\n")
            
            f.write("\\begin{tabular}{|l|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("\\rowcolor{gray!20} \\textbf{Class} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} \\\\ \\hline\n")
            
            for i, cls in enumerate(class_names):
                acc_cell = f"\\cellcolor{{lightblue}}\\textbf{{{class_metrics['Accuracy'][i]:.4f}}}" if i == best_acc_class else f"{class_metrics['Accuracy'][i]:.4f}"
                prec_cell = f"\\cellcolor{{lightgreen}}\\textbf{{{class_metrics['Precision'][i]:.4f}}}" if i == best_prec_class else f"{class_metrics['Precision'][i]:.4f}"
                recall_cell = f"\\cellcolor{{lightorange}}\\textbf{{{class_metrics['Recall'][i]:.4f}}}" if i == best_recall_class else f"{class_metrics['Recall'][i]:.4f}"
                
                f.write(f"{cls} & {acc_cell} & {prec_cell} & {recall_cell} \\\\ \\hline\n")
            
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    print(f"Created enhanced tables for {kernel} kernel")

print("Enhanced tables created successfully!")
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the file into Unbalanced and Balanced sections based on the dividers
    sections = re.split(r'={50}\n RESULTS — WITH CLASS WEIGHTING \(BALANCED\)\n={50}', content)
    unbalanced_text = sections[0]
    balanced_text = sections[1] if len(sections) > 1 else ""

    def extract_models(text):
        models = {}
        # Regex to capture Model Name, TP, TN, FP, FN, F1, Balanced Accuracy, and Best Params
        pattern = r'\[([^\]]+)\]\n.*?\n.*?\n.*?\n\s+TP: (\d+)\n\s+TN: (\d+)\n\s+FP: (\d+)\n\s+FN: (\d+)\n\s+F1 Score: ([\d.]+)\n\s+Balanced Accuracy: ([\d.]+)\n\s+Best Params: (\{.*?\})'
        
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            name = match.group(1).strip()
            models[name] = {
                'TP': int(match.group(2)),
                'TN': int(match.group(3)),
                'FP': int(match.group(4)),
                'FN': int(match.group(5)),
                'F1': float(match.group(6)),
                'Bal_Acc': float(match.group(7)),
                'Params': match.group(8)
            }
        return models

    unbalanced_data = extract_models(unbalanced_text)
    balanced_data = extract_models(balanced_text)

    # Combine both datasets into a single dictionary grouped by model
    combined = {}
    for model in unbalanced_data.keys():
        combined[model] = {
            'Unbalanced': unbalanced_data.get(model),
            'Balanced': balanced_data.get(model)
        }
    return combined

def plot_confusion_matrices(data):
    for model_name, stats in data.items():
        # Set up a 1x2 grid for side-by-side matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model_name} - Confusion Matrices', fontsize=16, fontweight='bold', y=1.05)

        for idx, weight_type in enumerate(['Unbalanced', 'Balanced']):
            ax = axes[idx]
            info = stats[weight_type]
            
            if not info:
                ax.axis('off')
                continue

            # Standard confusion matrix format: [[TN, FP], [FN, TP]]
            cm = np.array([[info['TN'], info['FP']], [info['FN'], info['TP']]])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        annot_kws={"size": 14},
                        xticklabels=['Pred Negative', 'Pred Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])

            # Add context and metrics to the title
            title_text = (f'{weight_type}\n'
                          f'F1 Score: {info["F1"]:.4f} | Balanced Acc: {info["Bal_Acc"]:.4f}')
            ax.set_title(title_text, fontsize=12, pad=15)

            # Place Best Params text below the individual subplot
            ax.text(0.5, -0.15, f"Best Params: {info['Params']}", 
                    ha='center', va='top', transform=ax.transAxes, 
                    fontsize=10, wrap=True, bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

        plt.tight_layout()
        # Adjust bottom to make room for the params text box
        plt.subplots_adjust(bottom=0.2) 
        
        # Save output image
        filename = f"{model_name.replace(' ', '_').lower()}_cm_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Generated: {filename}")
        plt.close()

if __name__ == "__main__":
    # Ensure 'res_4.txt' is in the same directory as this script
    filepath = 'res_4.txt'
    
    try:
        parsed_data = parse_results(filepath)
        if parsed_data:
            plot_confusion_matrices(parsed_data)
        else:
            print("No data extracted. Please check the text file format.")
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Make sure the file exists in the current directory.")
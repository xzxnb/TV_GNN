import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


ROOT_DIR = "./GNN/wfomi_data/json/color1_100k_mln/domain10"
TRAIN_SIZES = [100, 500, 1000, 5000, 20000]
TV_FOLDERS = [f"tv{round(i*0.1, 1)}" for i in range(11)]
TV_DISTANCE = [round(i*0.1, 1) for i in range(11)]
BAYES_VALUES = [round(0.5 + i*0.05, 2) for i in range(11)]
VAL_SIZE = 10000
OUTPUT_EXCEL = "val_accuracy_avg_results_with_bayes.xlsx"
OUTPUT_PLOT = "datasets_domain10_ave.png"


def extract_avg_val_accuracy(file_path):
    """Extract all val_accuracy values from a single JSON file and calculate the average."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        val_accuracies = []
        for item in data:
            if 'val_accuracy' in item:
                val_accuracies.append(item['val_accuracy'])

        if val_accuracies:
            return sum(val_accuracies) / len(val_accuracies)
        else:
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def create_excel_with_bayes():
    """Generate an Excel spreadsheet containing a bayes column."""
    df = pd.DataFrame(index=TRAIN_SIZES, columns=TV_FOLDERS)
    for tv_folder in TV_FOLDERS:
        tv_path = Path(ROOT_DIR) / tv_folder
        if not tv_path.exists():
            print(f"Warning: Folder {tv_path} does not exist, skip.")
            continue
        
        for train_size in TRAIN_SIZES:
            json_filename = f"train_size{train_size}_val_size{VAL_SIZE}.json"
            json_path = tv_path / json_filename
            avg_acc = extract_avg_val_accuracy(json_path)
            if avg_acc is not None:
                df.loc[train_size, tv_folder] = round(avg_acc, 6)
            else:
                df.loc[train_size, tv_folder] = np.nan

    df.loc["bayes"] = BAYES_VALUES
    df.to_excel(OUTPUT_EXCEL, index_label="train_num/TV")
    print(f"\nThe Excel file has been saved: {OUTPUT_EXCEL}")
    return df

def plot_excel_data(df):
    """Creating visual charts of Excel data"""
    train_data = df.drop("bayes")
    bayes_row = df.loc["bayes"]
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(TRAIN_SIZES)))
    for i, train_size in enumerate(TRAIN_SIZES):
        values = train_data.loc[train_size].dropna()
        ax.plot(TV_DISTANCE, values.values, 
                marker='o', linewidth=2, markersize=6,
                label=f"{train_size*2}", color=colors[i])

    ax.plot(TV_DISTANCE, bayes_row.values, 
            marker='s', linewidth=3, markersize=8, linestyle='--',
            label="Bayes", color='black', alpha=0.7)
    

    ax.set_xlabel("TV_distance", fontsize=26)
    ax.set_ylabel("average_accuracy", fontsize=26)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=30)
    
    ax.set_xticks(TV_DISTANCE)
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True, alpha=0.3, which='major', linestyle='-')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"The visualization chart has been saved: {OUTPUT_PLOT}")
    plt.show()

def main():
    df = create_excel_with_bayes()
    print("\nPreview of the generated table:")
    print(df.round(4))
    plot_excel_data(df)

if __name__ == "__main__":
    main()
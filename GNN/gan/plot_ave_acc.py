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
    """从单个JSON文件中提取所有val_accuracy并计算平均值"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取所有run的val_accuracy
        val_accuracies = []
        for item in data:
            if 'val_accuracy' in item:
                val_accuracies.append(item['val_accuracy'])
        
        # 计算平均值（如果有数据）
        if val_accuracies:
            return sum(val_accuracies) / len(val_accuracies)
        else:
            return None
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None

def create_excel_with_bayes():
    """生成包含bayes列的Excel表格"""
    df = pd.DataFrame(index=TRAIN_SIZES, columns=TV_FOLDERS)
    for tv_folder in TV_FOLDERS:
        tv_path = Path(ROOT_DIR) / tv_folder
        if not tv_path.exists():
            print(f"警告: 文件夹 {tv_path} 不存在，跳过")
            continue
        
        for train_size in TRAIN_SIZES:
            # 构建JSON文件路径
            json_filename = f"train_size{train_size}_val_size{VAL_SIZE}.json"
            json_path = tv_path / json_filename
            
            # 提取并计算平均值
            avg_acc = extract_avg_val_accuracy(json_path)
            if avg_acc is not None:
                df.loc[train_size, tv_folder] = round(avg_acc, 6)  # 保留6位小数
            else:
                df.loc[train_size, tv_folder] = np.nan  # 用NaN表示无数据
    
    # 新增bayes列：按tv列的顺序，对应0.5,0.55,...,1.0
    df.loc["bayes"] = BAYES_VALUES
    
    # 保存为Excel文件
    df.to_excel(OUTPUT_EXCEL, index_label="train_num/TV")
    print(f"\nExcel文件已保存: {OUTPUT_EXCEL}")
    return df

def plot_excel_data(df):
    """绘制Excel数据的可视化图表"""
    # 准备绘图数据：分离训练样本数和bayes行
    train_data = df.drop("bayes")  # 训练样本数的数据
    bayes_row = df.loc["bayes"]    # bayes列的数据
    
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制每个训练样本数的曲线
    colors = plt.cm.Set1(np.linspace(0, 1, len(TRAIN_SIZES)))
    for i, train_size in enumerate(TRAIN_SIZES):
        values = train_data.loc[train_size].dropna()  # 过滤NaN值
        ax.plot(TV_DISTANCE, values.values, 
                marker='o', linewidth=2, markersize=6,
                label=f"{train_size*2}", color=colors[i])
    
    # 绘制bayes的参考线（用虚线）
    ax.plot(TV_DISTANCE, bayes_row.values, 
            marker='s', linewidth=3, markersize=8, linestyle='--',
            label="Bayes", color='black', alpha=0.7)
    
    # 设置图表样式
    ax.set_title("TV-datasets", fontsize=18, pad=20)
    ax.set_xlabel("TV_distance", fontsize=18)
    ax.set_ylabel("average_accuracy", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=24)
    
    ax.set_xticks(TV_DISTANCE)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, which='major', linestyle='-')
    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45)
    
    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存: {OUTPUT_PLOT}")
    plt.show()

def main():
    df = create_excel_with_bayes()
    print("\n生成的表格预览:")
    print(df.round(4))  #
    plot_excel_data(df)

if __name__ == "__main__":
    main()
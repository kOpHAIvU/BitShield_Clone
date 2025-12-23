
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Tạo thư mục results nếu chưa có
os.makedirs('results', exist_ok=True)

# Dữ liệu gốc (Baseline)
# ResNet: MCC=75, TPR=71, F1=70
# Simple: MCC=71, TPR=58, F1=52

scenarios = ['Original'] + ['KB 1', 'KB 2', 'KB 3', 'KB 4']
models = ['ResNetSEBlockIoT', 'SimpleCNNIoT']

# Xây dựng dữ liệu thủ công để đảm bảo khớp
records = []

# 1. Original (Baseline)
records.append({'Scenario': 'Original', 'Model': 'ResNetSEBlockIoT', 'MCC': 75, 'TPR': 71, 'F1': 70})
records.append({'Scenario': 'Original', 'Model': 'SimpleCNNIoT', 'MCC': 71, 'TPR': 58, 'F1': 52})

# 2. Attack Scenarios (Data from table)
# KB1
records.append({'Scenario': 'KB 1', 'Model': 'ResNetSEBlockIoT', 'MCC': -10, 'TPR': 4, 'F1': 0})
records.append({'Scenario': 'KB 1', 'Model': 'SimpleCNNIoT', 'MCC': 66, 'TPR': 48, 'F1': 44})
# KB2
records.append({'Scenario': 'KB 2', 'Model': 'ResNetSEBlockIoT', 'MCC': 75, 'TPR': 71, 'F1': 70})
records.append({'Scenario': 'KB 2', 'Model': 'SimpleCNNIoT', 'MCC': 0, 'TPR': 4, 'F1': 0})
# KB3
records.append({'Scenario': 'KB 3', 'Model': 'ResNetSEBlockIoT', 'MCC': 0, 'TPR': 4, 'F1': 0})
records.append({'Scenario': 'KB 3', 'Model': 'SimpleCNNIoT', 'MCC': -2, 'TPR': 4, 'F1': 0})
# KB4
records.append({'Scenario': 'KB 4', 'Model': 'ResNetSEBlockIoT', 'MCC': 1, 'TPR': 4, 'F1': 1})
records.append({'Scenario': 'KB 4', 'Model': 'SimpleCNNIoT', 'MCC': -3, 'TPR': 0, 'F1': 0})

df = pd.DataFrame(records)

# Chuyển đổi dữ liệu sang dạng 'long'
df_melted = df.melt(id_vars=['Scenario', 'Model'], 
                    value_vars=['MCC', 'TPR', 'F1'],
                    var_name='Metric', value_name='Score')

# Chuyển đổi từ % sang scale -1 đến 1 (chia 100)
df_melted['Score'] = df_melted['Score'] / 100.0

# Metric names are already correct (MCC, TPR, F1) so no map needed, but rename F1 to F1-Score for display
df_melted['Metric'] = df_melted['Metric'].replace({'F1': 'F1-Score'})

# Thiết lập style
sns.set_theme(style="whitegrid", context="talk")

def plot_combined_rq1():
    plt.figure(figsize=(14, 8))
    
    g = sns.catplot(
        data=df_melted, 
        kind="bar",
        x="Scenario", 
        y="Score", 
        hue="Model",
        col="Metric",
        palette="ch:s=.25,rot=-.25",
        height=5, 
        aspect=0.8,
        edgecolor='black',
        order=['Original', 'KB 1', 'KB 2', 'KB 3', 'KB 4'] # Quy định thứ tự
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Tác động của BFA lên các mô hình (RQ1)', fontsize=20, weight='bold')
    
    # Định dạng nhãn trục Y và label trên cột
    for ax in g.axes.flat:
        ax.set_ylim(-0.2, 1.05) # MCC có thể âm
        ax.axhline(0, color='black', linewidth=1) # Đường 0 cho rõ
        
        for container in ax.containers:
            # Format 2 số thập phân (ví dụ 0.75)
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    
    g.set_axis_labels("Kịch bản tấn công", "Giá trị (Index)")
    
    output_file = 'results/rq1_impact_combined.png'
    g.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    plot_combined_rq1()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Tạo thư mục results nếu chưa có
os.makedirs('results', exist_ok=True)

# Dữ liệu từ hình ảnh bạn cung cấp
# Kịch bản 1, 2, 3, 4
data = {
    'Scenario': ['KB 1', 'KB 2', 'KB 3', 'KB 4'] * 2,
    'Model': ['ResNetSEBlockIoT'] * 4 + ['SimpleCNNIoT'] * 4,
    # Lấy giá trị "Sau" (After) để so sánh
    'MCC_After': [-10, 75, 0, 1, 66, 0, -2, -3], 
    'TPR_After': [4, 71, 4, 4, 48, 4, 4, 0],
    'F1_After': [0, 70, 0, 1, 44, 0, 0, 0]
}

df = pd.DataFrame(data)

# Chuyển đổi dữ liệu sang dạng 'long' để dễ vẽ với Seaborn
df_melted = df.melt(id_vars=['Scenario', 'Model'], 
                    value_vars=['MCC_After', 'TPR_After', 'F1_After'],
                    var_name='Metric', value_name='Score')

# Ánh xạ tên Metric cho đẹp
df_melted['Metric'] = df_melted['Metric'].map({
    'MCC_After': 'MCC',
    'TPR_After': 'TPR', 
    'F1_After': 'F1-Score'
})

# Thiết lập style
sns.set_theme(style="whitegrid", context="talk")

# Vẽ biểu đồ gộp (3 metrics x 4 kịch bản x 2 models) -> Có thể quá rối
# Thay vào đó vẽ 1 hình duy nhất cho MCC (chỉ số quan trọng nhất) hoặc vẽ 3 subplot

def plot_combined_rq1():
    plt.figure(figsize=(14, 8))
    
    # Vẽ biểu đồ cột
    # X trục: Kịch bản
    # Hue: Model
    # Row/Col: Metric
    
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
        edgecolor='black'
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Tác động của BFA lên các mô hình (RQ1)', fontsize=20, weight='bold')
    
    # Thêm baseline lines (vẽ thủ công hơi khó trên FacetGrid)
    # Thay vào đó chỉ hiển thị giá trị
    
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10)
    
    g.set_axis_labels("Kịch bản tấn công", "Giá trị sau tấn công (%)")
    
    output_file = 'results/rq1_impact_combined.png'
    g.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    plot_combined_rq1()

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_defense_results():
    """Load all defense results from JSON files"""
    results_dir = 'results/defense_results'
    data = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    # Parse filename for metadata
                    parts = filename.replace('.json', '').split('_')
                    dataset = parts[0]
                    model = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]
                    defense_type = parts[-2].upper()
                    
                    for entry in result.get('results', [result]):
                        entry['Dataset'] = dataset
                        entry['Model'] = model
                        entry['Defense Type'] = defense_type
                        data.append(entry)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return pd.DataFrame(data)


def visualize_cig_results():
    """Visualize CIG (Code Integrity Guard) results"""
    # Hardcoded data from experiments
    data = {
        'Noise Level': [0.1, 0.2, 0.5, 1.0],
        'ResNetSEBlockIoT': [86.1, 92.9, 96.7, 98.8],
        'SimpleCNNIoT': [90.0, 100.0, 100.0, 95.0]
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(data['Noise Level']))
    width = 0.35
    
    colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72'}
    
    bars1 = ax.bar(x - width/2, data['ResNetSEBlockIoT'], width, 
                   label='ResNetSEBlockIoT', color=colors['ResNetSEBlockIoT'],
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, data['SimpleCNNIoT'], width,
                   label='SimpleCNNIoT', color=colors['SimpleCNNIoT'],
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Cường độ nhiễu (Noise Level)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Tamper Fraction (%)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Bảng 5.4: CIG - Tỷ lệ tham số bị phát hiện thay đổi\n(IoTID20)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in data['Noise Level']], fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(title='Model', loc='lower right', fontsize=11, title_fontsize=12,
              framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1, label='100%')
    
    # Add annotation - positioned at upper right
    ax.annotate('CIG phát hiện thay đổi ở level trọng số (weight-level metric)',
               xy=(0.98, 0.98), xycoords='axes fraction',
               fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('results/fig_5_4_CIG_tamper_fraction.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: results/fig_5_4_CIG_tamper_fraction.png")
    plt.close()


def visualize_dig_results():
    """Visualize DIG (Data Integrity Guard) results"""
    # Hardcoded data from experiments
    data = {
        'Noise Level': [0.1, 0.2, 0.5, 1.0],
        'ResNetSEBlockIoT': [1.0, 1.0, 1.0, 1.0],
        'SimpleCNNIoT': [0.5, 0.5, 0.5, 0.5]
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(data['Noise Level']))
    width = 0.35
    
    colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00'}
    
    bars1 = ax.bar(x - width/2, data['ResNetSEBlockIoT'], width, 
                   label='ResNetSEBlockIoT', color=colors['ResNetSEBlockIoT'],
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, data['SimpleCNNIoT'], width,
                   label='SimpleCNNIoT', color=colors['SimpleCNNIoT'],
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Cường độ nhiễu (Noise Level)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate - TPR (%)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Bảng 5.5: DIG - Tỷ lệ input bị đánh dấu suspicious\n(IoTID20)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in data['Noise Level']], fontsize=12)
    ax.set_ylim(0, 10)
    ax.legend(title='Model', loc='upper left', fontsize=11, title_fontsize=12,
              framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add annotation - positioned at top right, legend is at upper left
    ax.annotate('DIG phát hiện ở level input (sample-level metric)\n'
                'TPR < 1%: DIG không hiệu quả với tấn công noise ngẫu nhiên',
               xy=(0.98, 0.95), xycoords='axes fraction',
               fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('results/fig_5_5_DIG_tpr.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: results/fig_5_5_DIG_tpr.png")
    plt.close()


def visualize_comparison_side_by_side():
    """Create side-by-side comparison of CIG and DIG"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    noise_levels = [0.1, 0.2, 0.5, 1.0]
    x = np.arange(len(noise_levels))
    width = 0.35
    
    # CIG data
    cig_data = {
        'ResNetSEBlockIoT': [86.1, 92.9, 96.7, 98.8],
        'SimpleCNNIoT': [90.0, 100.0, 100.0, 95.0]
    }
    
    # DIG data
    dig_data = {
        'ResNetSEBlockIoT': [1.0, 1.0, 1.0, 1.0],
        'SimpleCNNIoT': [0.5, 0.5, 0.5, 0.5]
    }
    
    # CIG subplot
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, cig_data['ResNetSEBlockIoT'], width, 
                    label='ResNetSEBlockIoT', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, cig_data['SimpleCNNIoT'], width,
                    label='SimpleCNNIoT', color='#A23B72')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Cường độ nhiễu', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Tamper Fraction (%)', fontsize=13, fontweight='bold')
    ax1.set_title('CIG: Phát hiện thay đổi trọng số\n(Weight-level)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in noise_levels])
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    # DIG subplot  
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, dig_data['ResNetSEBlockIoT'], width, 
                    label='ResNetSEBlockIoT', color='#E94F37')
    bars4 = ax2.bar(x + width/2, dig_data['SimpleCNNIoT'], width,
                    label='SimpleCNNIoT', color='#F77F00')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Cường độ nhiễu', fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('DIG: Phát hiện input suspicious\n(Sample-level)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in noise_levels])
    ax2.set_ylim(0, 10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('So sánh CIG và DIG trên IoTID20\n(Lưu ý: Hai metric khác nhau, không so sánh trực tiếp được)', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig_comparison_CIG_vs_DIG.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: results/fig_comparison_CIG_vs_DIG.png")
    plt.close()


def visualize_obfus_effectiveness():
    """Visualize OBFUS defense effectiveness"""
    # Data from experiments
    datasets = ['WUSTL', 'IoTID20', 'CICIoT2023']
    
    resnet_data = {
        'Baseline': [1.00, 0.86, 0.71],
        'After Attack': [0.46, 0.41, 0.08],
        'With OBFUS': [1.00, 0.79, 0.55]
    }
    
    simple_data = {
        'Baseline': [0.99, 0.78, 0.73],
        'After Attack': [-0.44, -0.18, -0.02],
        'With OBFUS': [0.02, -0.13, 0.07]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    # ResNet subplot
    ax1 = axes[0]
    bars1 = ax1.bar(x - width, resnet_data['Baseline'], width, label='Baseline', color='#2ECC71')
    bars2 = ax1.bar(x, resnet_data['After Attack'], width, label='After Attack', color='#E74C3C')
    bars3 = ax1.bar(x + width, resnet_data['With OBFUS'], width, label='With OBFUS', color='#3498DB')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax1.set_ylabel('MCC', fontsize=13, fontweight='bold')
    ax1.set_title('ResNetSEBlockIoT: OBFUS Recovery', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylim(-0.6, 1.2)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # SimpleCNN subplot
    ax2 = axes[1]
    bars4 = ax2.bar(x - width, simple_data['Baseline'], width, label='Baseline', color='#2ECC71')
    bars5 = ax2.bar(x, simple_data['After Attack'], width, label='After Attack', color='#E74C3C')
    bars6 = ax2.bar(x + width, simple_data['With OBFUS'], width, label='With OBFUS', color='#3498DB')
    
    for bars in [bars4, bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            offset = 3 if height >= 0 else -12
            va = 'bottom' if height >= 0 else 'top'
            ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset), textcoords="offset points",
                        ha='center', va=va, fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MCC', fontsize=13, fontweight='bold')
    ax2.set_title('SimpleCNNIoT: OBFUS Recovery', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylim(-0.6, 1.2)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.suptitle('OBFUS Defense Effectiveness: MCC Recovery (25 bit-flips, reseed mỗi 10 inferences)', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig_OBFUS_effectiveness.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: results/fig_OBFUS_effectiveness.png")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("BitShield Defense Visualization")
    print("="*60)
    
    print("\n--- Generating CIG Chart ---")
    visualize_cig_results()
    
    print("\n--- Generating DIG Chart ---")
    visualize_dig_results()
    
    print("\n--- Generating CIG vs DIG Comparison ---")
    visualize_comparison_side_by_side()
    
    print("\n--- Generating OBFUS Effectiveness Chart ---")
    visualize_obfus_effectiveness()
    
    print("\n" + "="*60)
    print("All visualizations saved to results/ folder")
    print("="*60)

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_all_defense_results():
    """Load all defense results from JSON files in results/defense_results"""
    results_dir = 'results/defense_results'
    
    cig_data = defaultdict(lambda: defaultdict(list))
    dig_data = defaultdict(lambda: defaultdict(list))
    obfus_data = []
    
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return cig_data, dig_data, obfus_data
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # Parse filename to get metadata
        # Format: Dataset_Model_defensetype_attack.json
        parts = filename.replace('.json', '').split('_')
        dataset = parts[0]
        
        if 'cig' in filename.lower():
            defense_type = 'CIG'
            model = '_'.join(parts[1:-2])
        elif 'dig' in filename.lower():
            defense_type = 'DIG'
            model = '_'.join(parts[1:-2])
        elif 'obfus' in filename.lower():
            defense_type = 'OBFUS'
            model = '_'.join(parts[1:-2])
        else:
            continue
        
        # Extract results
        if 'results' in data:
            results = data['results']
        else:
            results = [data]
        
        for entry in results:
            noise_level = entry.get('noise_level', entry.get('attack_strength', 0))
            detection_rate = entry.get('detection_rate', 0)
            
            if defense_type == 'CIG':
                cig_data[(dataset, model)][noise_level].append(detection_rate)
            elif defense_type == 'DIG':
                dig_data[(dataset, model)][noise_level].append(detection_rate)
            elif defense_type == 'OBFUS':
                obfus_data.append({
                    'dataset': dataset,
                    'model': model,
                    'baseline': entry.get('baseline_mcc', entry.get('baseline', 0)),
                    'attacked': entry.get('attacked_mcc', entry.get('after_attack', 0)),
                    'protected': entry.get('protected_mcc', entry.get('with_obfus', 0))
                })
    
    return cig_data, dig_data, obfus_data


def visualize_cig_from_data():
    """Visualize CIG results from actual data files"""
    cig_data, _, _ = load_all_defense_results()
    
    if not cig_data:
        print("No CIG data found. Using placeholder values.")
        return
    
    # Group by dataset
    datasets = set(k[0] for k in cig_data.keys())
    
    for dataset in datasets:
        models = [k[1] for k in cig_data.keys() if k[0] == dataset]
        if not models:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72', 
                  'PureCNN': '#A23B72', 'ResNet': '#2E86AB'}
        
        # Get all noise levels
        all_noise_levels = set()
        for model in models:
            all_noise_levels.update(cig_data[(dataset, model)].keys())
        noise_levels = sorted(all_noise_levels)
        
        if not noise_levels:
            continue
            
        x = np.arange(len(noise_levels))
        width = 0.35 if len(models) <= 2 else 0.25
        
        for i, model in enumerate(models):
            values = []
            for nl in noise_levels:
                rates = cig_data[(dataset, model)].get(nl, [0])
                values.append(np.mean(rates) if rates else 0)
            
            offset = width * (i - len(models)/2 + 0.5)
            color = colors.get(model, f'C{i}')
            bars = ax.bar(x + offset, values, width, label=model, 
                         color=color, edgecolor='white', linewidth=1.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Cường độ nhiễu (Noise Level)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Tamper Fraction (%)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(f'CIG - Tỷ lệ tham số bị phát hiện thay đổi\n({dataset})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in noise_levels], fontsize=12)
        ax.set_ylim(0, 115)
        ax.legend(title='Model', loc='lower right', fontsize=11, title_fontsize=12, framealpha=0.9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.annotate('CIG phát hiện thay đổi ở level trọng số (weight-level metric)',
                   xy=(0.98, 0.98), xycoords='axes fraction',
                   fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        output_file = f'results/fig_CIG_{dataset}_tamper_fraction.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def visualize_dig_from_data():
    """Visualize DIG results from actual data files"""
    _, dig_data, _ = load_all_defense_results()
    
    if not dig_data:
        print("No DIG data found.")
        return
    
    # Group by dataset
    datasets = set(k[0] for k in dig_data.keys())
    
    for dataset in datasets:
        models = [k[1] for k in dig_data.keys() if k[0] == dataset]
        if not models:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00',
                  'PureCNN': '#F77F00', 'ResNet': '#E94F37'}
        
        # Get all noise levels
        all_noise_levels = set()
        for model in models:
            all_noise_levels.update(dig_data[(dataset, model)].keys())
        noise_levels = sorted(all_noise_levels)
        
        if not noise_levels:
            continue
            
        x = np.arange(len(noise_levels))
        width = 0.35 if len(models) <= 2 else 0.25
        
        max_val = 0
        for i, model in enumerate(models):
            values = []
            for nl in noise_levels:
                rates = dig_data[(dataset, model)].get(nl, [0])
                val = np.mean(rates) if rates else 0
                values.append(val)
                max_val = max(max_val, val)
            
            offset = width * (i - len(models)/2 + 0.5)
            color = colors.get(model, f'C{i}')
            bars = ax.bar(x + offset, values, width, label=model, 
                         color=color, edgecolor='white', linewidth=1.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Cường độ nhiễu (Noise Level)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Positive Rate - TPR (%)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(f'DIG - Tỷ lệ input bị đánh dấu suspicious\n({dataset})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in noise_levels], fontsize=12)
        ax.set_ylim(0, max(10, max_val * 1.5))
        ax.legend(title='Model', loc='upper left', fontsize=11, title_fontsize=12, framealpha=0.9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        ax.annotate('DIG phát hiện ở level input (sample-level metric)\n'
                    'TPR thấp: DIG không hiệu quả với tấn công noise ngẫu nhiên',
                   xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        output_file = f'results/fig_DIG_{dataset}_tpr.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def visualize_combined_comparison():
    """Create side-by-side comparison of CIG vs DIG for all datasets"""
    cig_data, dig_data, _ = load_all_defense_results()
    
    if not cig_data and not dig_data:
        print("No data found for comparison.")
        return
    
    # Get common datasets
    cig_datasets = set(k[0] for k in cig_data.keys())
    dig_datasets = set(k[0] for k in dig_data.keys())
    common_datasets = cig_datasets & dig_datasets
    
    for dataset in common_datasets:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # CIG subplot
        ax1 = axes[0]
        cig_models = [k[1] for k in cig_data.keys() if k[0] == dataset]
        cig_colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72',
                      'PureCNN': '#A23B72'}
        
        all_noise = set()
        for m in cig_models:
            all_noise.update(cig_data[(dataset, m)].keys())
        noise_levels = sorted(all_noise)
        
        if noise_levels:
            x = np.arange(len(noise_levels))
            width = 0.35
            
            for i, model in enumerate(cig_models):
                values = [np.mean(cig_data[(dataset, model)].get(nl, [0])) for nl in noise_levels]
                offset = width * (i - len(cig_models)/2 + 0.5)
                bars = ax1.bar(x + offset, values, width, label=model, 
                              color=cig_colors.get(model, f'C{i}'))
                for bar, val in zip(bars, values):
                    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax1.set_xlabel('Cường độ nhiễu', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Tamper Fraction (%)', fontsize=13, fontweight='bold')
            ax1.set_title('CIG: Phát hiện thay đổi trọng số\n(Weight-level)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(n) for n in noise_levels])
            ax1.set_ylim(0, 115)
            ax1.legend(loc='lower right', fontsize=10)
            ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5)
        
        # DIG subplot
        ax2 = axes[1]
        dig_models = [k[1] for k in dig_data.keys() if k[0] == dataset]
        dig_colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00',
                      'PureCNN': '#F77F00'}
        
        all_noise_dig = set()
        for m in dig_models:
            all_noise_dig.update(dig_data[(dataset, m)].keys())
        noise_levels_dig = sorted(all_noise_dig)
        
        if noise_levels_dig:
            x2 = np.arange(len(noise_levels_dig))
            max_dig = 0
            
            for i, model in enumerate(dig_models):
                values = [np.mean(dig_data[(dataset, model)].get(nl, [0])) for nl in noise_levels_dig]
                max_dig = max(max_dig, max(values) if values else 0)
                offset = width * (i - len(dig_models)/2 + 0.5)
                bars = ax2.bar(x2 + offset, values, width, label=model,
                              color=dig_colors.get(model, f'C{i}'))
                for bar, val in zip(bars, values):
                    ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax2.set_xlabel('Cường độ nhiễu', fontsize=13, fontweight='bold')
            ax2.set_ylabel('True Positive Rate (%)', fontsize=13, fontweight='bold')
            ax2.set_title('DIG: Phát hiện input suspicious\n(Sample-level)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels([str(n) for n in noise_levels_dig])
            ax2.set_ylim(0, max(10, max_dig * 1.5))
            ax2.legend(loc='upper right', fontsize=10)
            ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
        
        plt.suptitle(f'So sánh CIG và DIG trên {dataset}\n(Lưu ý: Hai metric khác nhau, không so sánh trực tiếp được)', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = f'results/fig_comparison_CIG_vs_DIG_{dataset}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def print_summary():
    """Print summary of loaded data"""
    cig_data, dig_data, obfus_data = load_all_defense_results()
    
    print("\n" + "="*60)
    print("DATA SUMMARY FROM JSON FILES")
    print("="*60)
    
    print(f"\nCIG Data: {len(cig_data)} dataset-model combinations")
    for key, values in cig_data.items():
        print(f"  {key[0]} - {key[1]}:")
        for nl, rates in sorted(values.items()):
            print(f"    Noise {nl}: {np.mean(rates):.1f}%")
    
    print(f"\nDIG Data: {len(dig_data)} dataset-model combinations")
    for key, values in dig_data.items():
        print(f"  {key[0]} - {key[1]}:")
        for nl, rates in sorted(values.items()):
            print(f"    Noise {nl}: {np.mean(rates):.1f}%")
    
    print(f"\nOBFUS Data: {len(obfus_data)} entries")


if __name__ == "__main__":
    print("="*60)
    print("BitShield Defense Visualization (from real data)")
    print("="*60)
    
    print_summary()
    
    print("\n--- Generating CIG Charts ---")
    visualize_cig_from_data()
    
    print("\n--- Generating DIG Charts ---")
    visualize_dig_from_data()
    
    print("\n--- Generating Comparison Charts ---")
    visualize_combined_comparison()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)

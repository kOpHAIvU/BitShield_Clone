"""
Script to:
1. Combine all defense JSON results into combined_metrics.xlsx
2. Visualize CIG and DIG results from the combined file
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = 'results/defense_results'
COMBINED_FILE = 'results/combined_metrics.xlsx'


def create_combined_metrics():
    """Read all JSON files and create combined_metrics.xlsx"""
    all_data = []
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
        return False
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # Parse filename: Dataset_Model_defensetype_attack.json
        parts = filename.replace('.json', '').split('_')
        dataset = parts[0]
        
        if 'cig' in filename.lower():
            defense_type = 'CIG'
        elif 'dig' in filename.lower():
            defense_type = 'DIG'
        elif 'obfus' in filename.lower():
            defense_type = 'OBFUS'
        else:
            defense_type = 'Unknown'
        
        # Extract model name
        model_parts = []
        for p in parts[1:]:
            if p.lower() in ['cig', 'dig', 'obfus', 'attack', 'combined']:
                break
            model_parts.append(p)
        model = '_'.join(model_parts) if model_parts else 'Unknown'
        
        # Process results
        if 'results' in data:
            results = data['results']
        else:
            results = [data]
        
        for entry in results:
            row = {
                'Dataset': dataset,
                'Model': model,
                'Defense Type': defense_type,
                'Attack Strength': entry.get('noise_level', entry.get('attack_strength', 0)),
                'Detection Rate': entry.get('detection_rate', 0),
            }
            
            # Add optional fields
            for key in ['accuracy_before', 'accuracy_after', 'mcc_before', 'mcc_after',
                        'baseline_mcc', 'attacked_mcc', 'protected_mcc']:
                if key in entry:
                    row[key] = entry[key]
            
            all_data.append(row)
    
    if not all_data:
        print("No data found in JSON files")
        return False
    
    df = pd.DataFrame(all_data)
    df.to_excel(COMBINED_FILE, index=False)
    print(f"✓ Created {COMBINED_FILE} with {len(df)} rows")
    print(f"  Datasets: {df['Dataset'].unique().tolist()}")
    print(f"  Models: {df['Model'].unique().tolist()}")
    print(f"  Defense Types: {df['Defense Type'].unique().tolist()}")
    return True


def load_combined_metrics():
    """Load data from combined_metrics.xlsx"""
    if not os.path.exists(COMBINED_FILE):
        print(f"File not found: {COMBINED_FILE}")
        print("Creating combined_metrics.xlsx from JSON files...")
        if not create_combined_metrics():
            return None
    
    try:
        df = pd.read_excel(COMBINED_FILE)
        print(f"Loaded {len(df)} rows from {COMBINED_FILE}")
        return df
    except Exception as e:
        print(f"Error reading {COMBINED_FILE}: {e}")
        return None


def visualize_cig():
    """Visualize CIG results from combined_metrics.xlsx"""
    df = load_combined_metrics()
    if df is None:
        return
    
    df_cig = df[df['Defense Type'] == 'CIG']
    if df_cig.empty:
        print("No CIG data found")
        return
    
    datasets = df_cig['Dataset'].unique()
    
    for dataset in datasets:
        df_ds = df_cig[df_cig['Dataset'] == dataset]
        models = df_ds['Model'].unique()
        
        if len(models) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72',
                  'PureCNN': '#A23B72', 'ResNet': '#2E86AB'}
        
        noise_levels = sorted(df_ds['Attack Strength'].unique())
        x = np.arange(len(noise_levels))
        width = 0.35 if len(models) <= 2 else 0.25
        
        for i, model in enumerate(models):
            df_model = df_ds[df_ds['Model'] == model]
            values = []
            for nl in noise_levels:
                val = df_model[df_model['Attack Strength'] == nl]['Detection Rate'].mean()
                values.append(val if pd.notna(val) else 0)
            
            offset = width * (i - len(models)/2 + 0.5)
            color = colors.get(model, f'C{i}')
            bars = ax.bar(x + offset, values, width, label=model,
                         color=color, edgecolor='white', linewidth=1.5)
            
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
        output_file = f'results/fig_CIG_{dataset}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def visualize_dig():
    """Visualize DIG results from combined_metrics.xlsx"""
    df = load_combined_metrics()
    if df is None:
        return
    
    df_dig = df[df['Defense Type'] == 'DIG']
    if df_dig.empty:
        print("No DIG data found")
        return
    
    datasets = df_dig['Dataset'].unique()
    
    for dataset in datasets:
        df_ds = df_dig[df_dig['Dataset'] == dataset]
        models = df_ds['Model'].unique()
        
        if len(models) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00',
                  'PureCNN': '#F77F00', 'ResNet': '#E94F37'}
        
        noise_levels = sorted(df_ds['Attack Strength'].unique())
        x = np.arange(len(noise_levels))
        width = 0.35 if len(models) <= 2 else 0.25
        
        max_val = 0
        for i, model in enumerate(models):
            df_model = df_ds[df_ds['Model'] == model]
            values = []
            for nl in noise_levels:
                val = df_model[df_model['Attack Strength'] == nl]['Detection Rate'].mean()
                val = val if pd.notna(val) else 0
                values.append(val)
                max_val = max(max_val, val)
            
            offset = width * (i - len(models)/2 + 0.5)
            color = colors.get(model, f'C{i}')
            bars = ax.bar(x + offset, values, width, label=model,
                         color=color, edgecolor='white', linewidth=1.5)
            
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
                    'TPR thấp: DIG không hiệu quả với tấn công noise',
                   xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        output_file = f'results/fig_DIG_{dataset}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def visualize_comparison():
    """Create side-by-side comparison chart from combined_metrics.xlsx"""
    df = load_combined_metrics()
    if df is None:
        return
    
    df_filtered = df[df['Defense Type'].isin(['CIG', 'DIG'])]
    if df_filtered.empty:
        print("No CIG/DIG data found")
        return
    
    datasets = df_filtered['Dataset'].unique()
    
    for dataset in datasets:
        df_ds = df_filtered[df_filtered['Dataset'] == dataset]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # CIG subplot
        ax1 = axes[0]
        df_cig = df_ds[df_ds['Defense Type'] == 'CIG']
        
        if not df_cig.empty:
            models_cig = df_cig['Model'].unique()
            noise_levels = sorted(df_cig['Attack Strength'].unique())
            x = np.arange(len(noise_levels))
            width = 0.35
            cig_colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72', 'PureCNN': '#A23B72'}
            
            for i, model in enumerate(models_cig):
                df_model = df_cig[df_cig['Model'] == model]
                values = [df_model[df_model['Attack Strength'] == nl]['Detection Rate'].mean() for nl in noise_levels]
                values = [v if pd.notna(v) else 0 for v in values]
                
                offset = width * (i - len(models_cig)/2 + 0.5)
                bars = ax1.bar(x + offset, values, width, label=model, color=cig_colors.get(model, f'C{i}'))
                
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
        df_dig = df_ds[df_ds['Defense Type'] == 'DIG']
        
        if not df_dig.empty:
            models_dig = df_dig['Model'].unique()
            noise_levels_dig = sorted(df_dig['Attack Strength'].unique())
            x2 = np.arange(len(noise_levels_dig))
            dig_colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00', 'PureCNN': '#F77F00'}
            
            max_val = 0
            for i, model in enumerate(models_dig):
                df_model = df_dig[df_dig['Model'] == model]
                values = [df_model[df_model['Attack Strength'] == nl]['Detection Rate'].mean() for nl in noise_levels_dig]
                values = [v if pd.notna(v) else 0 for v in values]
                max_val = max(max_val, max(values) if values else 0)
                
                offset = width * (i - len(models_dig)/2 + 0.5)
                bars = ax2.bar(x2 + offset, values, width, label=model, color=dig_colors.get(model, f'C{i}'))
                
                for bar, val in zip(bars, values):
                    ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax2.set_xlabel('Cường độ nhiễu', fontsize=13, fontweight='bold')
            ax2.set_ylabel('True Positive Rate (%)', fontsize=13, fontweight='bold')
            ax2.set_title('DIG: Phát hiện input suspicious\n(Sample-level)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels([str(n) for n in noise_levels_dig])
            ax2.set_ylim(0, max(10, max_val * 1.5))
            ax2.legend(loc='upper right', fontsize=10)
            ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
        
        plt.suptitle(f'So sánh CIG và DIG trên {dataset}\n(Lưu ý: Hai metric khác nhau, không so sánh trực tiếp được)',
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = f'results/fig_comparison_{dataset}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def print_data_summary():
    """Print summary of data in combined_metrics.xlsx"""
    df = load_combined_metrics()
    if df is None:
        return
    
    print("\n" + "="*60)
    print("DATA SUMMARY FROM combined_metrics.xlsx")
    print("="*60)
    
    for defense_type in df['Defense Type'].unique():
        print(f"\n{defense_type}:")
        df_def = df[df['Defense Type'] == defense_type]
        
        for dataset in df_def['Dataset'].unique():
            print(f"  Dataset: {dataset}")
            df_ds = df_def[df_def['Dataset'] == dataset]
            
            for model in df_ds['Model'].unique():
                print(f"    Model: {model}")
                df_model = df_ds[df_ds['Model'] == model]
                
                for noise in sorted(df_model['Attack Strength'].unique()):
                    rate = df_model[df_model['Attack Strength'] == noise]['Detection Rate'].mean()
                    print(f"      Noise {noise}: {rate:.1f}%")


if __name__ == "__main__":
    print("="*60)
    print("BitShield Defense Visualization")
    print("(Reading from combined_metrics.xlsx)")
    print("="*60)
    
    # Step 1: Create combined_metrics.xlsx if not exists
    if not os.path.exists(COMBINED_FILE):
        print("\n--- Creating combined_metrics.xlsx ---")
        create_combined_metrics()
    
    # Step 2: Print summary
    print_data_summary()
    
    # Step 3: Generate visualizations
    print("\n--- Generating CIG Charts ---")
    visualize_cig()
    
    print("\n--- Generating DIG Charts ---")
    visualize_dig()
    
    print("\n--- Generating Comparison Charts ---")
    visualize_comparison()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)

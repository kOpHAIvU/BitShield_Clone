"""
BitShield Defense Visualization
- DIG: TPR (%) - Sample-level detection
- CIG: Layers Detected (%) - Layer-level detection
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = 'results/defense_results'
COMBINED_XLSX = 'results/combined_metrics.xlsx'


def load_from_excel():
    """Load defense results from combined_metrics.xlsx"""
    results = {
        'dig': {'random_flip': defaultdict(lambda: defaultdict(list)),
                'pbs': defaultdict(lambda: defaultdict(list))},
        'cig': {'random_flip': defaultdict(lambda: defaultdict(list)),
                'pbs': defaultdict(lambda: defaultdict(list))}
    }
    
    if not os.path.exists(COMBINED_XLSX):
        print(f"Excel file not found: {COMBINED_XLSX}")
        return results, False
    
    try:
        df = pd.read_excel(COMBINED_XLSX)
        print(f"Loaded Excel: {len(df)} rows")
    except ImportError:
        print("openpyxl not installed. Trying CSV fallback...")
        return results, False
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return results, False
    
    for _, row in df.iterrows():
        defense_type = str(row.get('Defense Type', '')).lower()
        if 'dig' in defense_type:
            defense = 'dig'
        elif 'cig' in defense_type:
            defense = 'cig'
        else:
            continue
        
        attack_mode_raw = str(row.get('Attack Mode', '')).lower()
        if 'random' in attack_mode_raw:
            attack_mode = 'random_flip'
        elif 'pbs' in attack_mode_raw:
            attack_mode = 'pbs'
        else:
            continue
        
        dataset = str(row.get('Dataset', 'Unknown'))
        model = str(row.get('Model', 'Unknown'))
        iters = row.get('Iterations', row.get('Attack Strength', 25))
        iters = int(iters) if pd.notna(iters) else 25
        det_rate = row.get('Detection Rate', 0)
        det_rate = float(det_rate) if pd.notna(det_rate) else 0
        
        key = (dataset, model)
        results[defense][attack_mode][key][iters].append(det_rate)
    
    return results, True


def load_from_csv_iterlogs():
    """Load defense results from CSV iterlog files"""
    results = {
        'dig': {'random_flip': defaultdict(lambda: defaultdict(list)),
                'pbs': defaultdict(lambda: defaultdict(list))},
        'cig': {'random_flip': defaultdict(lambda: defaultdict(list)),
                'pbs': defaultdict(lambda: defaultdict(list))}
    }
    
    if not os.path.exists(RESULTS_DIR):
        return results, False
    
    csv_files = glob.glob(os.path.join(RESULTS_DIR, '*_iterlog.csv'))
    if not csv_files:
        return results, False
    
    print(f"Found {len(csv_files)} CSV files")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            continue
        
        parts = filename.replace('_iterlog.csv', '').split('_')
        dataset = parts[0]
        
        # Determine defense type
        is_cig = 'cig' in filename.lower()
        defense = 'cig' if is_cig else 'dig'
        
        # Determine attack mode
        if 'random' in filename.lower():
            attack_mode = 'random_flip'
        elif 'pbs' in filename.lower():
            attack_mode = 'pbs'
        elif 'mode' in df.columns and len(df) > 0:
            mode_val = str(df['mode'].iloc[0]).lower()
            if 'random' in mode_val:
                attack_mode = 'random_flip'
            elif 'pbs' in mode_val:
                attack_mode = 'pbs'
            else:
                continue
        else:
            continue
        
        # Extract model name
        model_parts = []
        for p in parts[1:]:
            if p.lower() in ['random', 'flip', 'pbs', 'cig', 'dig']:
                break
            model_parts.append(p)
        model = '_'.join(model_parts) if model_parts else 'Unknown'
        
        # Get last iteration's data
        last_row = df.iloc[-1]
        max_iter = int(last_row['iteration'])
        
        # Get detection rate column (different for DIG vs CIG)
        if is_cig and 'cig_detection_rate_iter' in df.columns:
            det_rate = float(last_row['cig_detection_rate_iter'])
        elif 'dig_detection_rate_iter' in df.columns:
            det_rate = float(last_row['dig_detection_rate_iter'])
        else:
            continue
        
        key = (dataset, model)
        results[defense][attack_mode][key][max_iter].append(det_rate)
        print(f"  {defense.upper()}/{attack_mode}: {dataset}/{model} @ {max_iter}it = {det_rate:.1f}%")
    
    return results, True


def load_all_results():
    """Load from both Excel and CSV, merging results"""
    excel_results, excel_ok = load_from_excel()
    csv_results, csv_ok = load_from_csv_iterlogs()
    
    if not excel_ok and not csv_ok:
        print("\n⚠️ No data found!")
        return excel_results
    
    merged = excel_results
    for defense in ['dig', 'cig']:
        for attack in ['random_flip', 'pbs']:
            for key, iters_data in csv_results[defense][attack].items():
                for iters, rates in iters_data.items():
                    merged[defense][attack][key][iters].extend(rates)
    
    return merged


def visualize_all():
    """Create visualizations from loaded data"""
    results = load_all_results()
    
    for defense in ['dig', 'cig']:
        for attack_mode in ['random_flip', 'pbs']:
            data = results[defense][attack_mode]
            
            if not data:
                print(f"No {defense.upper()} data for {attack_mode}")
                continue
            
            datasets = set(k[0] for k in data.keys())
            
            for dataset in datasets:
                models = [(k[0], k[1]) for k in data.keys() if k[0] == dataset]
                if not models:
                    continue
                
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Different colors and labels for DIG vs CIG
                if defense == 'dig':
                    colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00'}
                    ylabel = 'True Positive Rate - TPR (%)'
                    title_metric = 'Sample-level Detection'
                else:
                    colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72'}
                    ylabel = 'Layers Detected (%)'
                    title_metric = 'Layer-level Detection'
                
                all_iters = set()
                for key in models:
                    all_iters.update(data[key].keys())
                iterations = sorted(all_iters)
                
                if not iterations:
                    continue
                
                x = np.arange(len(iterations))
                width = 0.35 if len(models) <= 2 else 0.25
                
                max_val = 0
                for i, key in enumerate(models):
                    model = key[1]
                    values = []
                    for it in iterations:
                        rates = data[key].get(it, [0])
                        val = np.mean(rates) if rates else 0
                        values.append(val)
                        max_val = max(max_val, val)
                    
                    offset = width * (i - len(models)/2 + 0.5)
                    color = colors.get(model, f'C{i}')
                    bars = ax.bar(x + offset, values, width, label=model,
                                 color=color, edgecolor='white', linewidth=1.5)
                    
                    for bar, val in zip(bars, values):
                        ax.annotate(f'{val:.1f}%',
                                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   xytext=(0, 5), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                attack_label = "Random Bit-Flip" if attack_mode == 'random_flip' else "PBS"
                ax.set_xlabel('Số vòng lật bit (Bit-flip Iterations)', fontsize=14, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
                ax.set_title(f'{defense.upper()}: {attack_label} Attack ({title_metric})\n{dataset}',
                            fontsize=16, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels([str(it) for it in iterations], fontsize=12)
                
                # Set y-axis limit
                if defense == 'cig':
                    # For CIG, show up to 100% (max layers)
                    ax.set_ylim(0, max(110, max_val * 1.1))
                    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% layers')
                else:
                    ax.set_ylim(0, max(20, max_val * 1.3))
                
                ax.legend(title='Model', loc='upper left' if defense == 'dig' else 'lower right',
                         fontsize=11, title_fontsize=12, framealpha=0.9)
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)
                
                # Add annotation explaining metric
                if defense == 'cig':
                    note = f'CIG đếm số layer có thay đổi / tổng số layer\n(Với N bit-flips, tối đa N layer bị ảnh hưởng)'
                else:
                    note = 'DIG đếm số input bị đánh dấu suspicious'
                ax.annotate(note, xy=(0.02, 0.98), xycoords='axes fraction',
                           fontsize=9, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                output_file = f'results/fig_{defense.upper()}_{attack_mode}_{dataset}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Saved: {output_file}")
                plt.close()


def create_combined_chart():
    """Create a single chart comparing CIG and DIG side by side"""
    results = load_all_results()
    
    for attack_mode in ['random_flip', 'pbs']:
        cig_data = results['cig'][attack_mode]
        dig_data = results['dig'][attack_mode]
        
        if not cig_data and not dig_data:
            continue
        
        all_datasets = set(k[0] for k in cig_data.keys()) | set(k[0] for k in dig_data.keys())
        
        for dataset in all_datasets:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            attack_label = "Random Bit-Flip" if attack_mode == 'random_flip' else "PBS"
            
            # CIG (left)
            ax1 = axes[0]
            cig_models = [(k[0], k[1]) for k in cig_data.keys() if k[0] == dataset]
            if cig_models:
                all_iters = set()
                for key in cig_models:
                    all_iters.update(cig_data[key].keys())
                iterations = sorted(all_iters)
                
                if iterations:
                    x = np.arange(len(iterations))
                    width = 0.35
                    cig_colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72'}
                    
                    for i, key in enumerate(cig_models):
                        model = key[1]
                        values = [np.mean(cig_data[key].get(it, [0])) for it in iterations]
                        offset = width * (i - len(cig_models)/2 + 0.5)
                        bars = ax1.bar(x + offset, values, width, label=model,
                                      color=cig_colors.get(model, f'C{i}'))
                        for bar, val in zip(bars, values):
                            ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    ax1.set_xlabel('Số vòng lật bit', fontsize=13, fontweight='bold')
                    ax1.set_ylabel('Layers Detected (%)', fontsize=13, fontweight='bold')
                    ax1.set_title(f'CIG: {attack_label}\n(Layer-level)', fontsize=14, fontweight='bold')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels([str(it) for it in iterations])
                    ax1.set_ylim(0, 110)
                    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5)
                    ax1.legend(loc='lower right', fontsize=10)
                    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
            
            # DIG (right)
            ax2 = axes[1]
            dig_models = [(k[0], k[1]) for k in dig_data.keys() if k[0] == dataset]
            if dig_models:
                all_iters = set()
                for key in dig_models:
                    all_iters.update(dig_data[key].keys())
                iterations = sorted(all_iters)
                
                if iterations:
                    x = np.arange(len(iterations))
                    dig_colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00'}
                    
                    max_val = 0
                    for i, key in enumerate(dig_models):
                        model = key[1]
                        values = [np.mean(dig_data[key].get(it, [0])) for it in iterations]
                        max_val = max(max_val, max(values) if values else 0)
                        offset = width * (i - len(dig_models)/2 + 0.5)
                        bars = ax2.bar(x + offset, values, width, label=model,
                                      color=dig_colors.get(model, f'C{i}'))
                        for bar, val in zip(bars, values):
                            ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    ax2.set_xlabel('Số vòng lật bit', fontsize=13, fontweight='bold')
                    ax2.set_ylabel('True Positive Rate (%)', fontsize=13, fontweight='bold')
                    ax2.set_title(f'DIG: {attack_label}\n(Sample-level)', fontsize=14, fontweight='bold')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels([str(it) for it in iterations])
                    ax2.set_ylim(0, max(20, max_val * 1.3))
                    ax2.legend(loc='upper right', fontsize=10)
                    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
            
            plt.suptitle(f'So sánh CIG và DIG: {attack_label} trên {dataset}',
                        fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            output_file = f'results/fig_comparison_{attack_mode}_{dataset}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_file}")
            plt.close()


def print_summary():
    """Print data summary"""
    results = load_all_results()
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    has_data = False
    for defense in ['dig', 'cig']:
        for attack in ['random_flip', 'pbs']:
            data = results[defense][attack]
            if data:
                has_data = True
                metric = "Layers Detected" if defense == 'cig' else "TPR"
                print(f"\n{defense.upper()} - {attack} ({metric}):")
                for key, iters_data in data.items():
                    print(f"  {key[0]} / {key[1]}:")
                    for iters, rates in sorted(iters_data.items()):
                        print(f"    {iters} bit-flips: {np.mean(rates):.1f}%")
    
    if not has_data:
        print("\nNo Random Bit-Flip or PBS data found!")


if __name__ == "__main__":
    print("="*60)
    print("BitShield Defense Visualization")
    print("="*60)
    
    print_summary()
    
    print("\n--- Generating Individual Charts ---")
    visualize_all()
    
    print("\n--- Generating Comparison Charts ---")
    create_combined_chart()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

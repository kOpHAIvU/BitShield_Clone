"""
BitShield Defense Visualization
Reads results from:
1. CSV iterlog files (primary)
2. combined_metrics.xlsx (if available)
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
        print(f"Columns: {df.columns.tolist()}")
    except ImportError:
        print("openpyxl not installed. Trying CSV fallback...")
        return results, False
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return results, False
    
    # Print unique values for debugging
    if 'Attack Mode' in df.columns:
        print(f"Attack Modes: {df['Attack Mode'].unique()}")
    if 'Defense Type' in df.columns:
        print(f"Defense Types: {df['Defense Type'].unique()}")
    
    for _, row in df.iterrows():
        # Get defense type
        defense_type = str(row.get('Defense Type', '')).lower()
        if 'dig' in defense_type:
            defense = 'dig'
        elif 'cig' in defense_type:
            defense = 'cig'
        else:
            continue
        
        # Get attack mode
        attack_mode_raw = str(row.get('Attack Mode', '')).lower()
        if 'random' in attack_mode_raw:
            attack_mode = 'random_flip'
        elif 'pbs' in attack_mode_raw:
            attack_mode = 'pbs'
        else:
            continue  # Skip noise or other attacks
        
        dataset = str(row.get('Dataset', 'Unknown'))
        model = str(row.get('Model', 'Unknown'))
        
        # Get iterations (try different column names)
        iters = row.get('Attack Strength', row.get('Iterations', row.get('Attack Iters', 25)))
        iters = int(iters) if pd.notna(iters) else 25
        
        # Get detection rate
        det_rate = row.get('Detection Rate', row.get('DIG Detection Rate', 0))
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
        print(f"Directory not found: {RESULTS_DIR}")
        return results, False
    
    csv_files = glob.glob(os.path.join(RESULTS_DIR, '*_iterlog.csv'))
    if not csv_files:
        print("No CSV iterlog files found")
        return results, False
    
    print(f"Found {len(csv_files)} CSV files")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue
        
        # Parse filename
        parts = filename.replace('_iterlog.csv', '').split('_')
        dataset = parts[0]
        
        # Determine defense type
        defense = 'cig' if 'cig' in filename.lower() else 'dig'
        
        # Determine attack mode
        if 'random' in filename.lower():
            attack_mode = 'random_flip'
        elif 'pbs' in filename.lower():
            attack_mode = 'pbs'
        else:
            if 'mode' in df.columns and len(df) > 0:
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
        if 'dig_detection_rate_iter' in df.columns:
            last_row = df.iloc[-1]
            max_iter = int(last_row['iteration'])
            det_rate = float(last_row['dig_detection_rate_iter'])
            
            key = (dataset, model)
            results[defense][attack_mode][key][max_iter].append(det_rate)
            print(f"  {defense.upper()}/{attack_mode}: {dataset}/{model} @ {max_iter}it = {det_rate:.1f}%")
    
    return results, True


def load_all_results():
    """Load from both Excel and CSV, merging results"""
    # Try Excel first
    excel_results, excel_ok = load_from_excel()
    
    # Then try CSV
    csv_results, csv_ok = load_from_csv_iterlogs()
    
    if not excel_ok and not csv_ok:
        print("\n⚠️ No data found!")
        return excel_results  # Return empty structure
    
    # Merge results (CSV takes precedence if duplicate)
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
                
                if defense == 'dig':
                    colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00'}
                    ylabel = 'True Positive Rate - TPR (%)'
                    ylim_max = None  # Auto
                else:
                    colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72'}
                    ylabel = 'Tamper Fraction (%)'
                    ylim_max = 115
                
                all_iters = set()
                for key in models:
                    all_iters.update(data[key].keys())
                iterations = sorted(all_iters)
                
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
                ax.set_title(f'{defense.upper()} Detection: {attack_label} Attack\n({dataset})',
                            fontsize=16, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels([str(it) for it in iterations], fontsize=12)
                
                if ylim_max:
                    ax.set_ylim(0, ylim_max)
                else:
                    ax.set_ylim(0, max(20, max_val * 1.3))
                
                ax.legend(title='Model', loc='upper left' if defense == 'dig' else 'lower right',
                         fontsize=11, title_fontsize=12, framealpha=0.9)
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)
                
                if defense == 'cig':
                    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                output_file = f'results/fig_{defense.upper()}_{attack_mode}_{dataset}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Saved: {output_file}")
                plt.close()


def print_summary():
    """Print data summary"""
    results = load_all_results()
    
    print("\n" + "="*60)
    print("FINAL DATA SUMMARY")
    print("="*60)
    
    has_data = False
    for defense in ['dig', 'cig']:
        for attack in ['random_flip', 'pbs']:
            data = results[defense][attack]
            if data:
                has_data = True
                print(f"\n{defense.upper()} - {attack}:")
                for key, iters_data in data.items():
                    print(f"  {key[0]} / {key[1]}:")
                    for iters, rates in sorted(iters_data.items()):
                        print(f"    {iters} iterations: {np.mean(rates):.1f}%")
    
    if not has_data:
        print("\n⚠️ No Random Bit-Flip or PBS data found!")
        print("Run experiments with --attack-mode random_flip or --attack-mode pbs")


if __name__ == "__main__":
    print("="*60)
    print("BitShield Defense Visualization")
    print("="*60)
    
    print_summary()
    
    print("\n--- Generating Charts ---")
    visualize_all()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

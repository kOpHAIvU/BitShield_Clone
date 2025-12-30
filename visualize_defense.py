"""
BitShield Defense Visualization
Reads results from Random Bit-Flip and PBS attacks for CIG/DIG evaluation
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = 'results/defense_results'


def load_defense_results():
    """Load all defense results from JSON files"""
    results = {
        'dig': {'random_flip': defaultdict(lambda: defaultdict(list)),
                'pbs': defaultdict(lambda: defaultdict(list))},
        'cig': {'random_flip': defaultdict(lambda: defaultdict(list)),
                'pbs': defaultdict(lambda: defaultdict(list))}
    }
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
        return results
    
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
        
        # Parse filename: Dataset_Model_defense_attack.json
        filename_lower = filename.lower()
        
        # Determine defense type
        if 'dig' in filename_lower:
            defense_type = 'dig'
        elif 'cig' in filename_lower:
            defense_type = 'cig'
        else:
            continue
        
        # Determine attack mode from data or filename
        attack_mode = None
        if 'attack_mode' in data:
            attack_mode = data['attack_mode']
        elif 'random_flip' in filename_lower:
            attack_mode = 'random_flip'
        elif 'pbs' in filename_lower:
            attack_mode = 'pbs'
        
        # Get from results if available
        if 'results' in data and len(data['results']) > 0:
            first_result = data['results'][0]
            if 'mode' in first_result:
                attack_mode = first_result['mode']
        
        if attack_mode not in ['random_flip', 'pbs']:
            continue
        
        # Parse dataset and model from filename
        parts = filename.replace('.json', '').split('_')
        dataset = parts[0]
        model_parts = []
        for p in parts[1:]:
            if p.lower() in ['dig', 'cig', 'attack', 'combined', 'random', 'flip', 'pbs']:
                break
            model_parts.append(p)
        model = '_'.join(model_parts) if model_parts else 'Unknown'
        
        # Process results
        entries = data.get('results', [data])
        for entry in entries:
            iters = entry.get('attack_iters', entry.get('iterations', 25))
            detection_rate = entry.get('detection_rate', 0)
            
            key = (dataset, model)
            results[defense_type][attack_mode][key][iters].append(detection_rate)
    
    return results


def visualize_dig_attacks():
    """Visualize DIG results for Random Bit-Flip and PBS attacks"""
    results = load_defense_results()
    dig_data = results['dig']
    
    for attack_mode in ['random_flip', 'pbs']:
        data = dig_data[attack_mode]
        
        if not data:
            print(f"No DIG data for {attack_mode}")
            continue
        
        # Group by dataset
        datasets = set(k[0] for k in data.keys())
        
        for dataset in datasets:
            models = [(k[0], k[1]) for k in data.keys() if k[0] == dataset]
            if not models:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00',
                      'PureCNN': '#F77F00', 'ResNet': '#E94F37'}
            
            # Get all iteration counts
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
                    height = bar.get_height()
                    ax.annotate(f'{val:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5), textcoords="offset points",
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            attack_label = "Random Bit-Flip" if attack_mode == 'random_flip' else "PBS"
            ax.set_xlabel('Số vòng lật bit (Bit-flip Iterations)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel('True Positive Rate - TPR (%)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_title(f'DIG Detection: {attack_label} Attack\n({dataset})',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([str(it) for it in iterations], fontsize=12)
            ax.set_ylim(0, max(20, max_val * 1.3))
            ax.legend(title='Model', loc='upper left', fontsize=11, title_fontsize=12, framealpha=0.9)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            output_file = f'results/fig_DIG_{attack_mode}_{dataset}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_file}")
            plt.close()


def visualize_cig_attacks():
    """Visualize CIG results for Random Bit-Flip and PBS attacks"""
    results = load_defense_results()
    cig_data = results['cig']
    
    for attack_mode in ['random_flip', 'pbs']:
        data = cig_data[attack_mode]
        
        if not data:
            print(f"No CIG data for {attack_mode}")
            continue
        
        # Group by dataset
        datasets = set(k[0] for k in data.keys())
        
        for dataset in datasets:
            models = [(k[0], k[1]) for k in data.keys() if k[0] == dataset]
            if not models:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = {'ResNetSEBlockIoT': '#2E86AB', 'SimpleCNNIoT': '#A23B72',
                      'PureCNN': '#A23B72', 'ResNet': '#2E86AB'}
            
            # Get all iteration counts
            all_iters = set()
            for key in models:
                all_iters.update(data[key].keys())
            iterations = sorted(all_iters)
            
            if not iterations:
                continue
            
            x = np.arange(len(iterations))
            width = 0.35 if len(models) <= 2 else 0.25
            
            for i, key in enumerate(models):
                model = key[1]
                values = []
                for it in iterations:
                    rates = data[key].get(it, [0])
                    val = np.mean(rates) if rates else 0
                    values.append(val)
                
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
            
            attack_label = "Random Bit-Flip" if attack_mode == 'random_flip' else "PBS"
            ax.set_xlabel('Số vòng lật bit (Bit-flip Iterations)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel('Tamper Fraction (%)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_title(f'CIG Detection: {attack_label} Attack\n({dataset})',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([str(it) for it in iterations], fontsize=12)
            ax.set_ylim(0, 115)
            ax.legend(title='Model', loc='lower right', fontsize=11, title_fontsize=12, framealpha=0.9)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1)
            
            plt.tight_layout()
            output_file = f'results/fig_CIG_{attack_mode}_{dataset}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_file}")
            plt.close()


def visualize_comparison():
    """Create side-by-side comparison of CIG vs DIG for each attack mode"""
    results = load_defense_results()
    
    for attack_mode in ['random_flip', 'pbs']:
        cig_data = results['cig'][attack_mode]
        dig_data = results['dig'][attack_mode]
        
        if not cig_data and not dig_data:
            continue
        
        # Get common datasets
        cig_datasets = set(k[0] for k in cig_data.keys())
        dig_datasets = set(k[0] for k in dig_data.keys())
        all_datasets = cig_datasets | dig_datasets
        
        for dataset in all_datasets:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            attack_label = "Random Bit-Flip" if attack_mode == 'random_flip' else "PBS"
            
            # CIG subplot
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
                    ax1.set_ylabel('Tamper Fraction (%)', fontsize=13, fontweight='bold')
                    ax1.set_title(f'CIG: {attack_label}\n(Weight-level)', fontsize=14, fontweight='bold')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels([str(it) for it in iterations])
                    ax1.set_ylim(0, 115)
                    ax1.legend(loc='lower right', fontsize=10)
                    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
            
            # DIG subplot
            ax2 = axes[1]
            dig_models = [(k[0], k[1]) for k in dig_data.keys() if k[0] == dataset]
            
            if dig_models:
                all_iters_dig = set()
                for key in dig_models:
                    all_iters_dig.update(dig_data[key].keys())
                iterations_dig = sorted(all_iters_dig)
                
                if iterations_dig:
                    x2 = np.arange(len(iterations_dig))
                    dig_colors = {'ResNetSEBlockIoT': '#E94F37', 'SimpleCNNIoT': '#F77F00'}
                    
                    max_val = 0
                    for i, key in enumerate(dig_models):
                        model = key[1]
                        values = [np.mean(dig_data[key].get(it, [0])) for it in iterations_dig]
                        max_val = max(max_val, max(values) if values else 0)
                        offset = width * (i - len(dig_models)/2 + 0.5)
                        bars = ax2.bar(x2 + offset, values, width, label=model,
                                      color=dig_colors.get(model, f'C{i}'))
                        for bar, val in zip(bars, values):
                            ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    ax2.set_xlabel('Số vòng lật bit', fontsize=13, fontweight='bold')
                    ax2.set_ylabel('True Positive Rate (%)', fontsize=13, fontweight='bold')
                    ax2.set_title(f'DIG: {attack_label}\n(Sample-level)', fontsize=14, fontweight='bold')
                    ax2.set_xticks(x2)
                    ax2.set_xticklabels([str(it) for it in iterations_dig])
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


def print_data_summary():
    """Print summary of loaded data"""
    results = load_defense_results()
    
    print("\n" + "="*60)
    print("DATA SUMMARY (Random Bit-Flip & PBS Attacks)")
    print("="*60)
    
    for defense in ['dig', 'cig']:
        for attack_mode in ['random_flip', 'pbs']:
            data = results[defense][attack_mode]
            if not data:
                continue
            
            print(f"\n{defense.upper()} - {attack_mode}:")
            for key, iters_data in data.items():
                print(f"  {key[0]} - {key[1]}:")
                for iters, rates in sorted(iters_data.items()):
                    print(f"    {iters} iters: {np.mean(rates):.1f}%")


if __name__ == "__main__":
    print("="*60)
    print("BitShield Defense Visualization")
    print("(Random Bit-Flip & PBS Attacks)")
    print("="*60)
    
    print_data_summary()
    
    print("\n--- Generating DIG Charts ---")
    visualize_dig_attacks()
    
    print("\n--- Generating CIG Charts ---")
    visualize_cig_attacks()
    
    print("\n--- Generating Comparison Charts ---")
    visualize_comparison()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)

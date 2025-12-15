#!/usr/bin/env python3
"""
Visualize OBFUS experiment results
Creates comparison charts for 3 stages across 4 attack modes
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def load_results(json_file: str) -> Dict:
    """Load experiment results from JSON"""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_metrics_by_stage(results: Dict, attack_modes: List[str]) -> Dict:
    """
    Extract metrics organized by stage and attack mode
    
    Returns:
        {
            'accuracy': {
                'baseline': [...],
                'no_defense': [...],
                'with_obfus': [...]
            },
            ...
        }
    """
    metrics_names = ['accuracy', 'f1', 'tpr', 'mcc']
    data = {metric: {'baseline': [], 'no_defense': [], 'with_obfus': []} for metric in metrics_names}
    
    baseline = results['baseline']
    
    for metric in metrics_names:
        # Baseline (same value for all attack modes)
        baseline_value = baseline.get(metric, 0.0)
        data[metric]['baseline'] = [baseline_value] * len(attack_modes)
        
        # Attack without defense
        for mode in attack_modes:
            if mode in results['attack_no_defense']:
                value = results['attack_no_defense'][mode].get(metric, 0.0)
            else:
                value = 0.0
            data[metric]['no_defense'].append(value)
        
        # Attack with OBFUS
        for mode in attack_modes:
            if mode in results.get('attack_with_obfus', {}):
                value = results['attack_with_obfus'][mode].get(metric, 0.0)
            else:
                value = 0.0
            data[metric]['with_obfus'].append(value)
    
    return data

def plot_metric_comparison(data: Dict, metric_name: str, attack_modes: List[str],
                           title: str, output_file: str):
    """Plot comparison for a single metric across all stages and attack modes"""
    
    x = np.arange(len(attack_modes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    baseline_vals = data[metric_name]['baseline']
    no_defense_vals = data[metric_name]['no_defense']
    with_obfus_vals = data[metric_name]['with_obfus']
    
    # Create bars
    bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline (No Attack)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, no_defense_vals, width, label='Attack (No Defense)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, with_obfus_vals, width, label='Attack (With OBFUS)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    # Customize
    ax.set_xlabel('Attack Mode', fontsize=13, fontweight='bold')
    ax.set_ylabel(metric_name.upper(), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace('2', '→') for m in attack_modes], fontsize=11)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at baseline
    baseline_mean = np.mean(baseline_vals)
    ax.axhline(y=baseline_mean, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline Level')
    
    plt.tight_layout()
    ensure_dir_of(output_file)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_all_metrics_combined(data: Dict, attack_modes: List[str],
                              dataset_name: str, model_name: str, output_file: str):
    """Plot all 4 metrics in a 2x2 grid"""
    
    metrics = ['accuracy', 'f1', 'tpr', 'mcc']
    metric_labels = ['Accuracy', 'F1-Score', 'TPR (Recall)', 'MCC']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'OBFUS Defense Comparison: {model_name} on {dataset_name}',
                fontsize=18, fontweight='bold', y=0.995)
    
    x = np.arange(len(attack_modes))
    width = 0.25
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        baseline_vals = data[metric]['baseline']
        no_defense_vals = data[metric]['no_defense']
        with_obfus_vals = data[metric]['with_obfus']
        
        # Create bars
        bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline',
                      color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, no_defense_vals, width, label='No Defense',
                      color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + width, with_obfus_vals, width, label='With OBFUS',
                      color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)
        
        # Add value labels
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        # Customize
        ax.set_xlabel('Attack Mode', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace('2', '→') for m in attack_modes], fontsize=10, rotation=15, ha='right')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add baseline reference line
        baseline_mean = np.mean(baseline_vals)
        ax.axhline(y=baseline_mean, color='green', linestyle='--', linewidth=1, alpha=0.4)
    
    plt.tight_layout()
    ensure_dir_of(output_file)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_defense_effectiveness(data: Dict, attack_modes: List[str],
                               dataset_name: str, model_name: str, output_file: str):
    """
    Plot defense effectiveness: how much OBFUS recovers performance
    compared to no defense
    """
    
    metrics = ['accuracy', 'f1', 'tpr', 'mcc']
    metric_labels = ['Accuracy', 'F1-Score', 'TPR', 'MCC']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(attack_modes))
    width = 0.2
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        baseline_vals = np.array(data[metric]['baseline'])
        no_defense_vals = np.array(data[metric]['no_defense'])
        with_obfus_vals = np.array(data[metric]['with_obfus'])
        
        # Calculate degradation and recovery
        degradation = baseline_vals - no_defense_vals  # How much performance lost
        recovery = with_obfus_vals - no_defense_vals  # How much OBFUS recovers
        recovery_percent = (recovery / (degradation + 1e-9)) * 100  # % of lost performance recovered
        
        # Plot recovery percentage
        ax.bar(x + idx * width, recovery_percent, width, label=label, alpha=0.8, edgecolor='black')
    
    # Customize
    ax.set_xlabel('Attack Mode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance Recovery (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'OBFUS Defense Effectiveness: {model_name} on {dataset_name}\n'
                f'(% of Lost Performance Recovered)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper().replace('2', '→') for m in attack_modes], fontsize=11)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.7, label='100% Recovery')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No Recovery')
    
    plt.tight_layout()
    ensure_dir_of(output_file)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_degradation_comparison(data: Dict, attack_modes: List[str],
                                dataset_name: str, model_name: str, output_file: str):
    """
    Plot performance degradation from baseline for each stage
    """
    
    metrics = ['accuracy', 'f1', 'tpr', 'mcc']
    metric_labels = ['Accuracy', 'F1-Score', 'TPR', 'MCC']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Performance Degradation Analysis: {model_name} on {dataset_name}',
                fontsize=18, fontweight='bold', y=0.995)
    
    x = np.arange(len(attack_modes))
    width = 0.35
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        baseline_vals = np.array(data[metric]['baseline'])
        no_defense_vals = np.array(data[metric]['no_defense'])
        with_obfus_vals = np.array(data[metric]['with_obfus'])
        
        # Calculate degradation (negative values = performance drop)
        degradation_no_def = no_defense_vals - baseline_vals
        degradation_with_obfus = with_obfus_vals - baseline_vals
        
        # Create bars
        bars1 = ax.bar(x - width/2, degradation_no_def, width, label='No Defense',
                      color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, degradation_with_obfus, width, label='With OBFUS',
                      color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)
        
        # Add value labels
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                label_y = height
                va = 'bottom' if height >= 0 else 'top'
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3 if height >= 0 else -3),
                           textcoords="offset points",
                           ha='center', va=va,
                           fontsize=8, fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        
        # Customize
        ax.set_xlabel('Attack Mode', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Δ {label} (from Baseline)', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Degradation', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace('2', '→') for m in attack_modes], fontsize=10, rotation=15, ha='right')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    ensure_dir_of(output_file)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_summary_table(results: Dict, attack_modes: List[str]) -> str:
    """Create a markdown summary table"""
    
    lines = []
    lines.append("# OBFUS Defense Experiment Summary\n")
    lines.append(f"**Model:** {results['model']}\n")
    lines.append(f"**Dataset:** {results['dataset']}\n")
    lines.append(f"**Attack Iterations:** {results['attack_iters']}\n\n")
    
    # Baseline
    lines.append("## Baseline (No Attack)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    baseline = results['baseline']
    lines.append(f"| Accuracy | {baseline['accuracy']:.4f} |")
    lines.append(f"| F1-Score | {baseline['f1']:.4f} |")
    lines.append(f"| TPR | {baseline['tpr']:.4f} |")
    lines.append(f"| MCC | {baseline['mcc']:.4f} |")
    lines.append("\n")
    
    # Attack without defense
    lines.append("## Attack Without Defense\n")
    lines.append("| Attack Mode | Accuracy | F1-Score | TPR | MCC |")
    lines.append("|-------------|----------|----------|-----|-----|")
    for mode in attack_modes:
        if mode in results['attack_no_defense']:
            m = results['attack_no_defense'][mode]
            lines.append(f"| {mode.upper()} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['tpr']:.4f} | {m['mcc']:.4f} |")
    lines.append("\n")
    
    # Attack with OBFUS
    if 'attack_with_obfus' in results and results['attack_with_obfus']:
        lines.append("## Attack With OBFUS Defense\n")
        lines.append("| Attack Mode | Accuracy | F1-Score | TPR | MCC |")
        lines.append("|-------------|----------|----------|-----|-----|")
        for mode in attack_modes:
            if mode in results['attack_with_obfus']:
                m = results['attack_with_obfus'][mode]
                lines.append(f"| {mode.upper()} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['tpr']:.4f} | {m['mcc']:.4f} |")
        lines.append("\n")
        
        # Performance recovery
        lines.append("## Defense Effectiveness (Performance Recovery)\n")
        lines.append("| Attack Mode | Acc Recovery | F1 Recovery | TPR Recovery | MCC Recovery |")
        lines.append("|-------------|--------------|-------------|--------------|--------------|")
        for mode in attack_modes:
            if mode in results['attack_with_obfus'] and mode in results['attack_no_defense']:
                no_def = results['attack_no_defense'][mode]
                with_obfus = results['attack_with_obfus'][mode]
                baseline = results['baseline']
                
                acc_deg = baseline['accuracy'] - no_def['accuracy']
                acc_rec = with_obfus['accuracy'] - no_def['accuracy']
                acc_pct = (acc_rec / (acc_deg + 1e-9)) * 100
                
                f1_deg = baseline['f1'] - no_def['f1']
                f1_rec = with_obfus['f1'] - no_def['f1']
                f1_pct = (f1_rec / (f1_deg + 1e-9)) * 100
                
                tpr_deg = baseline['tpr'] - no_def['tpr']
                tpr_rec = with_obfus['tpr'] - no_def['tpr']
                tpr_pct = (tpr_rec / (tpr_deg + 1e-9)) * 100
                
                mcc_deg = baseline['mcc'] - no_def['mcc']
                mcc_rec = with_obfus['mcc'] - no_def['mcc']
                mcc_pct = (mcc_rec / (mcc_deg + 1e-9)) * 100
                
                lines.append(f"| {mode.upper()} | {acc_pct:.1f}% | {f1_pct:.1f}% | {tpr_pct:.1f}% | {mcc_pct:.1f}% |")
    
    return "\n".join(lines)

def visualize_results(json_file: str, output_dir: str = None):
    """Main function to create all visualizations"""
    
    # Load results
    results = load_results(json_file)
    
    dataset_name = results['dataset']
    model_name = results['model']
    
    # Determine output directory
    if output_dir is None:
        output_dir = f'results/obfus_visualizations/{dataset_name}_{model_name}'
    
    print(f"\n{'='*60}")
    print(f"Creating visualizations for {model_name} on {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Get attack modes
    attack_modes = list(results['attack_no_defense'].keys())
    
    # Extract data
    data = extract_metrics_by_stage(results, attack_modes)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Individual metric plots
    for metric in ['accuracy', 'f1', 'tpr', 'mcc']:
        plot_metric_comparison(
            data, metric, attack_modes,
            title=f'{metric.upper()} Comparison: {model_name} on {dataset_name}',
            output_file=f'{output_dir}/{metric}_comparison.png'
        )
    
    # 2. All metrics combined
    plot_all_metrics_combined(
        data, attack_modes, dataset_name, model_name,
        output_file=f'{output_dir}/all_metrics_combined.png'
    )
    
    # 3. Defense effectiveness
    if 'attack_with_obfus' in results and results['attack_with_obfus']:
        plot_defense_effectiveness(
            data, attack_modes, dataset_name, model_name,
            output_file=f'{output_dir}/defense_effectiveness.png'
        )
        
        # 4. Degradation comparison
        plot_degradation_comparison(
            data, attack_modes, dataset_name, model_name,
            output_file=f'{output_dir}/degradation_comparison.png'
        )
    
    # 5. Create summary table
    summary_md = create_summary_table(results, attack_modes)
    summary_file = f'{output_dir}/summary.md'
    ensure_dir_of(summary_file)
    with open(summary_file, 'w') as f:
        f.write(summary_md)
    print(f"Saved: {summary_file}")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize OBFUS experiment results')
    parser.add_argument('json_file', type=str, help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_results(args.json_file, args.output_dir)


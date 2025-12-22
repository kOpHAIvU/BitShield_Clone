#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a simple benchmark overview diagram using matplotlib
No graphviz dependency required - works out of the box
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os


def create_simple_overview():
    """Create simple benchmark overview using matplotlib"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Benchmark Overview: BFA on AI-NIDS với OBFUS Defense', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Define colors
    color_input = '#E3F2FD'
    color_preprocess = '#FFF8E1'
    color_baseline = '#C8E6C9'
    color_attack = '#FFEBEE'
    color_defense = '#E8EAF6'
    color_metrics = '#F1F8E9'
    color_output = '#EDE7F6'
    
    # Row 1: Input (y=8.0)
    y_input = 8.0
    boxes_input = [
        (0.5, y_input, 2.5, 0.8, 'Datasets\n• IoTID20\n• WUSTL-IIoT-2021\n• CICIoT2023', color_input),
        (3.5, y_input, 2.5, 0.8, 'Model Architectures\n• ResNetSEBlockIoT\n• SimpleCNNIoT\n• EfficientCNN', color_input),
        (6.5, y_input, 2.5, 0.8, 'Configurations\n• Attack Modes\n• Attack Iterations\n• OBFUS Params', color_input),
    ]
    
    # Row 2: Preprocessing (y=6.5)
    y_preprocess = 6.5
    boxes_preprocess = [
        (1.5, y_preprocess, 2.5, 0.8, 'Data Loading\n• Train/Test Split\n• Normalization', color_preprocess),
        (5.0, y_preprocess, 2.5, 0.8, 'Model Training\n• Train AI-NIDS\n• Quantization (8-bit)', color_preprocess),
    ]
    
    # Row 3: Benchmark Stages (y=4.5, 3.5, 2.5)
    y_stage1 = 4.8
    y_stage2 = 3.5
    y_stage3 = 2.2
    boxes_stages = [
        (1.5, y_stage1, 6.5, 0.6, 'Stage 1: BASELINE\nOriginal Model • No Attack • Clean Performance', color_baseline),
        (1.5, y_stage2, 6.5, 0.6, 'Stage 2: ATTACK (No Defense)\n4 Attack Modes: PBS, Random, PBS→Random, Random→PBS', color_attack),
        (1.5, y_stage3, 6.5, 0.6, 'Stage 3: ATTACK + OBFUS\nSame 4 Attack Modes WITH OBFUS Defense', color_defense),
    ]
    
    # Row 4: Metrics & Output (y=0.8)
    y_bottom = 0.8
    boxes_bottom = [
        (0.5, y_bottom, 2.5, 0.6, 'Performance Metrics\n• Accuracy, F1\n• TPR, MCC', color_metrics),
        (3.5, y_bottom, 2.5, 0.6, 'Attack Impact\n• Accuracy Drop\n• Degradation Rate', color_metrics),
        (6.5, y_bottom, 2.5, 0.6, 'Results & Viz\n• JSON Reports\n• Charts', color_output),
    ]
    
    # Draw all boxes
    all_boxes = boxes_input + boxes_preprocess + boxes_stages + boxes_bottom
    for x, y, w, h, text, color in all_boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=9, multialignment='center')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    
    # Input to Preprocessing
    ax.annotate('', xy=(2.75, y_preprocess+0.8), xytext=(1.75, y_input),
               arrowprops=arrow_props)
    ax.annotate('', xy=(6.25, y_preprocess+0.8), xytext=(4.75, y_input),
               arrowprops=arrow_props)
    
    # Preprocessing to Stage 1
    ax.annotate('', xy=(3.0, y_stage1+0.6), xytext=(2.75, y_preprocess),
               arrowprops=arrow_props)
    ax.annotate('', xy=(6.0, y_stage1+0.6), xytext=(6.25, y_preprocess),
               arrowprops=arrow_props)
    
    # Stage 1 -> Stage 2 -> Stage 3
    ax.annotate('', xy=(4.75, y_stage2+0.6), xytext=(4.75, y_stage1),
               arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, y_stage3+0.6), xytext=(4.75, y_stage2),
               arrowprops=arrow_props)
    
    # Stages to Metrics
    ax.annotate('', xy=(1.75, y_bottom+0.6), xytext=(3.0, y_stage3),
               arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, y_bottom+0.6), xytext=(4.75, y_stage3),
               arrowprops=arrow_props)
    ax.annotate('', xy=(7.75, y_bottom+0.6), xytext=(6.5, y_stage3),
               arrowprops=arrow_props)
    
    plt.tight_layout()
    return fig


def create_three_stages_detail():
    """Create detailed 3-stages diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(9, 7.5, 'Three Stages Benchmark Detail', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Colors
    color_baseline = '#C8E6C9'
    color_attack = '#FFEBEE'
    color_defense = '#E8EAF6'
    
    # Stage 1: Baseline (x=1-5)
    x1, w1 = 1, 4
    boxes_s1 = [
        (x1, 5.5, w1, 0.7, 'Load Trained Model\nClean Quantized Weights', color_baseline),
        (x1, 4.5, w1, 0.7, 'Evaluate on Test Set\nNo Attack Applied', color_baseline),
        (x1, 3.5, w1, 0.7, 'Baseline Metrics\nAcc, F1, TPR, MCC', color_baseline),
    ]
    ax.text(x1 + w1/2, 6.5, 'Stage 1: BASELINE', ha='center', fontsize=12, fontweight='bold')
    
    # Stage 2: Attack (x=6-11)
    x2, w2 = 6, 5
    boxes_s2 = [
        (x2, 5.5, w2, 0.7, 'For Each Attack Mode\nPBS, Random, PBS→Random, Random→PBS', color_attack),
        (x2, 4.5, w2, 0.7, 'Apply Bit Flips\nN iterations (e.g., 25)', color_attack),
        (x2, 3.5, w2, 0.7, 'Model Degradation\nWeights Corrupted', color_attack),
        (x2, 2.5, w2, 0.7, 'Measure Impact\nAccuracy Drop, Performance Loss', color_attack),
    ]
    ax.text(x2 + w2/2, 6.5, 'Stage 2: ATTACK (No Defense)', ha='center', fontsize=12, fontweight='bold')
    
    # Stage 3: OBFUS (x=12-17)
    x3, w3 = 12, 5
    boxes_s3 = [
        (x3, 5.9, w3, 0.5, 'Initialize OBFUS', color_defense),
        (x3, 5.2, w3, 0.5, 'Baseline with OBFUS', color_defense),
        (x3, 4.5, w3, 0.5, 'For Each Attack Mode', color_defense),
        (x3, 3.8, w3, 0.5, 'Apply Bit Flips + Detection', color_defense),
        (x3, 3.1, w3, 0.5, 'OBFUS Detection', color_defense),
        (x3, 2.4, w3, 0.5, 'Model Recovery', color_defense),
        (x3, 1.7, w3, 0.5, 'Measure Defense', color_defense),
    ]
    ax.text(x3 + w3/2, 6.5, 'Stage 3: ATTACK + OBFUS', ha='center', fontsize=12, fontweight='bold')
    
    # Draw all boxes
    for boxes in [boxes_s1, boxes_s2, boxes_s3]:
        for x, y, w, h, text, color in boxes:
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=color, linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=9, multialignment='center')
    
    # Arrows within stages
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    
    # Stage 1 arrows
    for i in range(len(boxes_s1)-1):
        y_from = boxes_s1[i][1]
        y_to = boxes_s1[i+1][1] + boxes_s1[i+1][3]
        ax.annotate('', xy=(x1 + w1/2, y_to), xytext=(x1 + w1/2, y_from),
                   arrowprops=arrow_props)
    
    # Stage 2 arrows
    for i in range(len(boxes_s2)-1):
        y_from = boxes_s2[i][1]
        y_to = boxes_s2[i+1][1] + boxes_s2[i+1][3]
        ax.annotate('', xy=(x2 + w2/2, y_to), xytext=(x2 + w2/2, y_from),
                   arrowprops=arrow_props)
    
    # Stage 3 arrows
    for i in range(len(boxes_s3)-1):
        y_from = boxes_s3[i][1]
        y_to = boxes_s3[i+1][1] + boxes_s3[i+1][3]
        ax.annotate('', xy=(x3 + w3/2, y_to), xytext=(x3 + w3/2, y_from),
                   arrowprops=arrow_props)
    
    # Cross-stage connections (dashed)
    arrow_props_dashed = dict(arrowstyle='->', lw=1.2, color='gray', linestyle='dashed')
    
    # Stage 1 to Stage 2
    ax.annotate('', xy=(x2, 5.5 + 0.35), xytext=(x1 + w1, 3.5 + 0.35),
               arrowprops=arrow_props_dashed)
    ax.text((x1 + w1 + x2)/2, 4.7, 'Reference', ha='center', fontsize=8, color='gray')
    
    # Stage 1 to Stage 3
    ax.annotate('', xy=(x3, 5.2 + 0.25), xytext=(x1 + w1, 3.5 + 0.35),
               arrowprops=arrow_props_dashed)
    
    plt.tight_layout()
    return fig


def main():
    """Generate all simple diagrams"""
    output_dir = os.path.join(os.path.dirname(__file__), 'diagrams')
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Generating simple benchmark overview diagrams (matplotlib)...")
    
    # Generate diagrams
    diagrams = {
        'simple_overview': create_simple_overview(),
        'simple_three_stages': create_three_stages_detail(),
    }
    
    for name, fig in diagrams.items():
        output_path = os.path.join(output_dir, f'{name}.png')
        print(f"  [+] Generating {name}...")
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"      [OK] Saved: {output_path}")
        plt.close(fig)
    
    print(f"\n[SUCCESS] All simple diagrams generated!")
    print(f"[INFO] Output directory: {output_dir}")
    print("\nNote: These are simplified versions using matplotlib.")
    print("For higher quality diagrams, install graphviz and run generate_benchmark_diagram.py")


if __name__ == '__main__':
    main()

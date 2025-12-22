#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate high-quality benchmark overview diagram for research paper/presentation
Uses Graphviz to create professional-looking flowcharts
"""

import sys
import os

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    # Don't print here, will print in main()


def create_overall_benchmark_diagram():
    """Create overall benchmark system overview diagram"""
    dot = Digraph(comment='Benchmark Overview', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')

    # Input Stage
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='INPUT', style='filled', color='lightblue', fontsize='12', fontname='Arial Bold')
        c.node('datasets', 'Datasets\n• IoTID20\n• WUSTL-IIoT-2021\n• CICIoT2023', fillcolor='#E3F2FD')
        c.node('models', 'Model Architectures\n• ResNetSEBlockIoT\n• SimpleCNNIoT\n• EfficientCNN\n• PureCNN', fillcolor='#E3F2FD')
        c.node('configs', 'Configurations\n• Attack Modes\n• Attack Iterations\n• OBFUS Parameters', fillcolor='#E3F2FD')

    # Preprocessing
    with dot.subgraph(name='cluster_preprocess') as c:
        c.attr(label='PREPROCESSING', style='filled', color='lightyellow', fontsize='12', fontname='Arial Bold')
        c.node('data_loading', 'Data Loading\n• Train/Test/Val Split\n• Feature Engineering\n• Normalization', fillcolor='#FFF8E1')
        c.node('model_training', 'Model Training\n• Train AI-NIDS\n• Quantization (8-bit)\n• Save Checkpoints', fillcolor='#FFF8E1')

    # Benchmark Pipeline
    with dot.subgraph(name='cluster_pipeline') as c:
        c.attr(label='BENCHMARK PIPELINE', style='filled', color='lightgreen', fontsize='12', fontname='Arial Bold')
        c.node('stage1', 'Stage 1: BASELINE\n• No Attack\n• Clean Performance\n• Reference Metrics', fillcolor='#C8E6C9')
        c.node('stage2', 'Stage 2: ATTACK (No Defense)\n• 4 Attack Modes\n• PBS, Random\n• PBS→Random, Random→PBS', fillcolor='#FFEBEE')
        c.node('stage3', 'Stage 3: ATTACK + OBFUS\n• Same 4 Attack Modes\n• WITH OBFUS Defense\n• Detection & Recovery', fillcolor='#E8EAF6')

    # Metrics
    with dot.subgraph(name='cluster_metrics') as c:
        c.attr(label='METRICS COLLECTION', style='filled', color='#F1F8E9', fontsize='12', fontname='Arial Bold')
        c.node('perf_metrics', 'Performance Metrics\n• Accuracy\n• F1-Score\n• TPR, MCC', fillcolor='#F1F8E9')
        c.node('attack_metrics', 'Attack Impact\n• Accuracy Drop\n• Degradation Rate', fillcolor='#F1F8E9')
        c.node('defense_metrics', 'Defense Effectiveness\n• Detection Rate\n• False Positive\n• Recovery Success', fillcolor='#F1F8E9')

    # Output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='OUTPUT', style='filled', color='#EDE7F6', fontsize='12', fontname='Arial Bold')
        c.node('results', 'Results\n• JSON Reports\n• Statistical Analysis\n• Comparative Tables', fillcolor='#EDE7F6')
        c.node('viz', 'Visualizations\n• Performance Charts\n• Attack Impact Plots\n• Defense Comparison', fillcolor='#EDE7F6')

    # Edges
    dot.edge('datasets', 'data_loading')
    dot.edge('models', 'model_training')
    dot.edge('configs', 'stage1')
    
    dot.edge('data_loading', 'stage1')
    dot.edge('model_training', 'stage1')
    
    dot.edge('stage1', 'stage2')
    dot.edge('stage2', 'stage3')
    
    dot.edge('stage1', 'perf_metrics')
    dot.edge('stage2', 'attack_metrics')
    dot.edge('stage3', 'defense_metrics')
    
    dot.edge('perf_metrics', 'results')
    dot.edge('attack_metrics', 'results')
    dot.edge('defense_metrics', 'results')
    
    dot.edge('results', 'viz')

    return dot


def create_three_stages_diagram():
    """Create detailed 3-stages benchmark diagram"""
    dot = Digraph(comment='Three Stages Benchmark', format='png')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.5', ranksep='1.0')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Stage 1: Baseline
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Stage 1: BASELINE', style='filled', color='#C8E6C9', fontsize='12', fontname='Arial Bold')
        c.node('s1_load', 'Load Trained Model\nClean Quantized Weights', fillcolor='#C8E6C9')
        c.node('s1_eval', 'Evaluate on Test Set\nNo Attack Applied', fillcolor='#C8E6C9')
        c.node('s1_metrics', 'Baseline Metrics\n• Acc, F1, TPR, MCC\n• Class Distribution', fillcolor='#C8E6C9')
        c.edge('s1_load', 's1_eval')
        c.edge('s1_eval', 's1_metrics')

    # Stage 2: Attack (No Defense)
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Stage 2: ATTACK (No Defense)', style='filled', color='#FFEBEE', fontsize='12', fontname='Arial Bold')
        c.node('s2_modes', 'For Each Attack Mode\n• PBS\n• Random\n• PBS→Random\n• Random→PBS', fillcolor='#FFEBEE')
        c.node('s2_flip', 'Apply Bit Flips\nN iterations\n(e.g., 25 flips)', fillcolor='#FFEBEE')
        c.node('s2_degrade', 'Model Degradation\nWeights Corrupted', fillcolor='#FFEBEE')
        c.node('s2_measure', 'Measure Impact\n• Accuracy Drop\n• Performance Loss', fillcolor='#FFEBEE')
        c.edge('s2_modes', 's2_flip')
        c.edge('s2_flip', 's2_degrade')
        c.edge('s2_degrade', 's2_measure')

    # Stage 3: Attack + OBFUS
    with dot.subgraph(name='cluster_stage3') as c:
        c.attr(label='Stage 3: ATTACK + OBFUS', style='filled', color='#E8EAF6', fontsize='12', fontname='Arial Bold')
        c.node('s3_init', 'Initialize OBFUS\n• Wrap Layers\n• Calibrate Monitors', fillcolor='#E8EAF6')
        c.node('s3_baseline', 'Baseline with OBFUS\nCheck No Degradation', fillcolor='#E8EAF6')
        c.node('s3_modes', 'For Each Attack Mode\nSame 4 Modes', fillcolor='#E8EAF6')
        c.node('s3_flip', 'Apply Bit Flips\n+ Periodic Detection', fillcolor='#E8EAF6')
        c.node('s3_detect', 'OBFUS Detection\n• SIG Monitor\n• Bit Fingerprint\n• Adaptive Reseed', fillcolor='#E8EAF6')
        c.node('s3_recover', 'Model Recovery\nWeights Restored', fillcolor='#E8EAF6')
        c.node('s3_measure', 'Measure Defense\n• Detection Rate\n• Accuracy Preserved', fillcolor='#E8EAF6')
        c.edge('s3_init', 's3_baseline')
        c.edge('s3_baseline', 's3_modes')
        c.edge('s3_modes', 's3_flip')
        c.edge('s3_flip', 's3_detect')
        c.edge('s3_detect', 's3_recover')
        c.edge('s3_recover', 's3_measure')

    # Cross-stage edges
    dot.edge('s1_metrics', 's2_modes', style='dashed', label='Reference')
    dot.edge('s1_metrics', 's3_init', style='dashed', label='Reference')
    dot.edge('s2_measure', 's3_measure', style='dashed', label='Compare', color='red')

    return dot


def create_bfa_mechanism_diagram():
    """Create Bit Flip Attack mechanism diagram"""
    dot = Digraph(comment='BFA Mechanism', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Target Model
    with dot.subgraph(name='cluster_target') as c:
        c.attr(label='TARGET: Quantized AI-NIDS Model', style='filled', color='#E3F2FD', fontsize='12', fontname='Arial Bold')
        c.node('model', 'AI-NIDS Model\nQuantized Layers (8-bit)\nConv1D + Linear', fillcolor='#E3F2FD')
        c.node('weights', 'Quantized Weights\nStored as 8-bit integers\nRange: [-128, 127]', fillcolor='#E3F2FD')
        c.edge('model', 'weights')

    # Attack Mode 1: PBS
    with dot.subgraph(name='cluster_pbs') as c:
        c.attr(label='Attack Mode 1: PBS (Progressive Bit Search)', style='filled', color='#FFEBEE', fontsize='12', fontname='Arial Bold')
        c.node('pbs_search', 'Search Strategy\n• Try random bit flips\n• Compute loss change\n• Keep best flip', fillcolor='#FFEBEE')
        c.node('pbs_select', 'Select Worst Flip\nMax loss increase\n= Max damage', fillcolor='#FFEBEE')
        c.node('pbs_apply', 'Apply Permanent Flip\nUpdate weight', fillcolor='#FFEBEE')
        c.edge('pbs_search', 'pbs_select')
        c.edge('pbs_select', 'pbs_apply')

    # Attack Mode 2: Random
    with dot.subgraph(name='cluster_random') as c:
        c.attr(label='Attack Mode 2: Random Bit Flip', style='filled', color='#FFF3E0', fontsize='12', fontname='Arial Bold')
        c.node('rand_select', 'Random Selection\n• Random layer\n• Random weight\n• Random bit (0-7)', fillcolor='#FFF3E0')
        c.node('rand_flip', 'Flip Bit\nXOR operation\nNo optimization', fillcolor='#FFF3E0')
        c.edge('rand_select', 'rand_flip')

    # Attack Mode 3: PBS→Random
    with dot.subgraph(name='cluster_p2r') as c:
        c.attr(label='Attack Mode 3: PBS→Random', style='filled', color='#F3E5F5', fontsize='12', fontname='Arial Bold')
        c.node('p2r_1', 'PBS First\nFind worst bit', fillcolor='#F3E5F5')
        c.node('p2r_2', 'Random After\nAdd noise', fillcolor='#F3E5F5')
        c.edge('p2r_1', 'p2r_2')

    # Attack Mode 4: Random→PBS
    with dot.subgraph(name='cluster_r2p') as c:
        c.attr(label='Attack Mode 4: Random→PBS', style='filled', color='#E0F2F1', fontsize='12', fontname='Arial Bold')
        c.node('r2p_1', 'Random First\nAdd noise', fillcolor='#E0F2F1')
        c.node('r2p_2', 'PBS After\nOptimize damage', fillcolor='#E0F2F1')
        c.edge('r2p_1', 'r2p_2')

    # Connect target to attacks
    dot.edge('weights', 'pbs_search')
    dot.edge('weights', 'rand_select')
    dot.edge('weights', 'p2r_1')
    dot.edge('weights', 'r2p_1')

    return dot


def create_obfus_defense_diagram():
    """Create OBFUS defense mechanism diagram"""
    dot = Digraph(comment='OBFUS Defense', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Components
    with dot.subgraph(name='cluster_components') as c:
        c.attr(label='OBFUS Components', style='filled', color='#E8EAF6', fontsize='12', fontname='Arial Bold')
        c.node('obfus_layer', 'Obfuscation Layer\nWrap critical layers\nwith ObfusPair', fillcolor='#E8EAF6')
        c.node('sig_monitor', 'SIG Monitor\n• KL Divergence\n• Gradient Norm\n• Detect anomalies', fillcolor='#E8EAF6')
        c.node('bit_fp', 'Bit Fingerprint\n• Track PSI\n• Monitor entropy drift', fillcolor='#E8EAF6')
        c.node('controller', 'Controller Policy\n• Fuse alerts\n• Trigger recovery', fillcolor='#E8EAF6')

    # Detection Flow
    with dot.subgraph(name='cluster_flow') as c:
        c.attr(label='Detection & Recovery Flow', style='filled', color='#C8E6C9', fontsize='12', fontname='Arial Bold')
        c.node('calibrate', 'Calibration Phase\n• Build baseline\n• Normal behavior profile', fillcolor='#C8E6C9')
        c.node('periodic', 'Periodic Checking\nEvery N iterations\n(e.g., every 20 steps)', fillcolor='#FFF8E1')
        c.node('monitor', 'Monitor Signals\n• SIG: KL + Grad\n• FP: PSI + Entropy', fillcolor='#FFF8E1')
        c.node('decision', 'Alert Triggered?', shape='diamond', fillcolor='#FFF3E0')
        c.node('reseed', 'Reseed Obfuscation\nRandomize layers', fillcolor='#FFEBEE')
        c.node('continue', 'Continue Inference\nNo action needed', fillcolor='#C8E6C9')
        
        c.edge('calibrate', 'periodic')
        c.edge('periodic', 'monitor')
        c.edge('monitor', 'decision')
        c.edge('decision', 'reseed', label='Yes')
        c.edge('decision', 'continue', label='No')
        c.edge('reseed', 'periodic', style='dashed')
        c.edge('continue', 'periodic', style='dashed')

    # Metrics
    with dot.subgraph(name='cluster_effectiveness') as c:
        c.attr(label='Defense Effectiveness', style='filled', color='#F1F8E9', fontsize='12', fontname='Arial Bold')
        c.node('det_rate', 'Detection Rate\n% of attacks detected', fillcolor='#F1F8E9')
        c.node('fp_rate', 'False Positive\nFalse alarms on clean data', fillcolor='#F1F8E9')
        c.node('recovery', 'Recovery Success\nModel performance restored', fillcolor='#F1F8E9')
        c.node('overhead', 'Performance Overhead\nLatency + Memory', fillcolor='#F1F8E9')

    # Connections
    dot.edge('obfus_layer', 'calibrate')
    dot.edge('sig_monitor', 'monitor')
    dot.edge('bit_fp', 'monitor')
    dot.edge('controller', 'decision')
    
    dot.edge('reseed', 'det_rate')
    dot.edge('continue', 'fp_rate')
    dot.edge('reseed', 'recovery')
    dot.edge('periodic', 'overhead')

    return dot


def main():
    """Generate all diagrams"""
    if not GRAPHVIZ_AVAILABLE:
        print("[ERROR] Cannot generate diagrams without graphviz.")
        print("Please install:")
        print("  1. Python package: pip install graphviz")
        print("  2. System graphviz: https://graphviz.org/download/")
        print("")
        print("Alternative: View the Mermaid diagrams in docs/BENCHMARK_OVERVIEW.md")
        return

    output_dir = os.path.join(os.path.dirname(__file__), 'diagrams')
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Generating benchmark overview diagrams...")

    # Generate diagrams
    diagrams = {
        'overall_benchmark': create_overall_benchmark_diagram(),
        'three_stages': create_three_stages_diagram(),
        'bfa_mechanism': create_bfa_mechanism_diagram(),
        'obfus_defense': create_obfus_defense_diagram(),
    }

    for name, dot in diagrams.items():
        output_path = os.path.join(output_dir, name)
        print(f"  [+] Generating {name}...")
        
        # Render as PNG
        dot.render(output_path, format='png', cleanup=True)
        print(f"      [OK] Saved: {output_path}.png")
        
        # Render as SVG (scalable for papers)
        dot.render(output_path, format='svg', cleanup=True)
        print(f"      [OK] Saved: {output_path}.svg")

    print(f"\n[SUCCESS] All diagrams generated successfully!")
    print(f"[INFO] Output directory: {output_dir}")
    print("\nUsage:")
    print("   - PNG files: For presentations, slides")
    print("   - SVG files: For papers, high-quality publications")


if __name__ == '__main__':
    main()

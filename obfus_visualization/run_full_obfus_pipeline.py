#!/usr/bin/env python3
"""
Full OBFUS experiment pipeline:
1. Run experiments (baseline, attack without defense, attack with OBFUS)
2. Generate visualizations
3. Create summary report
"""

import subprocess
import sys
import os
import argparse

def run_command(cmd):
    """Run command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError: Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run full OBFUS experiment pipeline')
    parser.add_argument('model', type=str, help='Model name (e.g., ResNetSEBlockIoT)')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CICIoT2023, IoTID20)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--attack-iters', type=int, default=25, help='Number of attack iterations')
    parser.add_argument('--attack-modes', type=str, default='pbs,random,pbs2random,random2pbs',
                       help='Comma-separated attack modes')
    
    # OBFUS options
    parser.add_argument('--sig-period', type=int, default=20, help='SIG probe period')
    parser.add_argument('--sig-k', type=float, default=3.0, help='SIG threshold k')
    parser.add_argument('--obfus-targets', type=str, default='linear,conv1d',
                       help='Comma-separated obfuscation targets')
    parser.add_argument('--obfus-max-layers', type=int, default=None,
                       help='Max layers to obfuscate (None=all)')
    parser.add_argument('--obfus-initial-reseed', action='store_true',
                       help='Reseed obfuscation layers initially')
    parser.add_argument('--obfus-auto-reseed', type=int, default=10,
                       help='Proactive reseed period')
    
    args = parser.parse_args()
    
    # Output file
    output_json = f'results/obfus_experiments/{args.dataset}_{args.model}_obfus_experiment.json'
    
    print(f"\n{'='*70}")
    print(f"  OBFUS EXPERIMENT PIPELINE")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Attack Iterations: {args.attack_iters}")
    print(f"Attack Modes: {args.attack_modes}")
    print(f"Output: {output_json}")
    print(f"{'='*70}\n")
    
    # Step 1: Run experiments
    print("\n" + "="*70)
    print("STEP 1: Running Experiments")
    print("="*70 + "\n")
    
    cmd = f"python run_obfus_experiments.py {args.model} {args.dataset} " \
          f"--device {args.device} " \
          f"--attack-iters {args.attack_iters} " \
          f"--attack-modes {args.attack_modes} " \
          f"--with-obfus " \
          f"--sig-period {args.sig_period} " \
          f"--sig-k {args.sig_k} " \
          f"--obfus-targets {args.obfus_targets} " \
          f"--obfus-auto-reseed {args.obfus_auto_reseed} " \
          f"--output {output_json}"
    
    if args.obfus_max_layers is not None:
        cmd += f" --obfus-max-layers {args.obfus_max_layers}"
    
    if args.obfus_initial_reseed:
        cmd += " --obfus-initial-reseed"
    
    run_command(cmd)
    
    # Step 2: Generate visualizations
    print("\n" + "="*70)
    print("STEP 2: Generating Visualizations")
    print("="*70 + "\n")
    
    cmd = f"python visualize_obfus_results.py {output_json}"
    run_command(cmd)
    
    # Done
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_json}")
    print(f"Visualizations saved to: results/obfus_visualizations/{args.dataset}_{args.model}/")
    print("\n")

if __name__ == '__main__':
    main()


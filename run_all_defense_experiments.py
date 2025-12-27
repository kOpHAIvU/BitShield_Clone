"""
Script to run all defense experiments for all models and datasets.
This will generate data for visualize_defense.py to create charts.
"""

import subprocess
import sys

def run_all_experiments():
    models = ['ResNetSEBlockIoT', 'SimpleCNNIoT']
    datasets = ['IoTID20', 'WUSTL', 'CICIoT2023']
    defense_modes = ['dig', 'cig']
    
    total = len(models) * len(datasets) * len(defense_modes)
    current = 0
    
    print("=" * 60)
    print("Running all defense experiments")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Defense modes: {defense_modes}")
    print(f"Total experiments: {total}")
    print("=" * 60)
    
    for model in models:
        for dataset in datasets:
            for mode in defense_modes:
                current += 1
                print(f"\n[{current}/{total}] Running: {mode} {model} {dataset}")
                print("-" * 40)
                
                cmd = [
                    sys.executable, 
                    'attack_with_defense_extended.py',
                    mode, model, dataset,
                    '--device', 'cuda'
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=False)
                    if result.returncode == 0:
                        print(f"✓ Completed: {mode} {model} {dataset}")
                    else:
                        print(f"✗ Failed: {mode} {model} {dataset}")
                except Exception as e:
                    print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("Now run: python visualize_defense.py")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()

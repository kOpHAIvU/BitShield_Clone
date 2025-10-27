#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False

def demo_training():
    """Demo training on all datasets"""
    print("🎯 DEMO: Training on Extended Datasets")
    print("="*80)
    
    datasets = ['IoTID20', 'WUSTL', 'CICIoT2023']
    models = ['ResNetSEBlockIoT', 'SimpleCNNIoT']
    
    for dataset in datasets:
        for model in models:
            cmd = f"python support/models/train_extended.py {model} {dataset} --epochs 5 --device cpu"
            success = run_command(cmd, f"Training {model} on {dataset}")
            if not success:
                print(f"⚠️  Training failed for {model} on {dataset}")
                break

def demo_defense():
    """Demo defense testing on all datasets"""
    print("🛡️ DEMO: Defense Testing on Extended Datasets")
    print("="*80)
    
    datasets = ['IoTID20', 'WUSTL', 'CICIoT2023']
    models = ['ResNetSEBlockIoT', 'SimpleCNNIoT']
    defenses = ['dig', 'cig', 'combined']
    
    for dataset in datasets:
        for model in models:
            for defense in defenses:
                cmd = f"python attack_with_defense_extended.py {defense} {model} {dataset} --device cpu"
                success = run_command(cmd, f"Testing {defense} defense for {model} on {dataset}")
                if not success:
                    print(f"⚠️  Defense testing failed for {defense} on {model} on {dataset}")

def demo_specific(dataset, model, epochs=5):
    """Demo specific dataset and model"""
    print(f"🎯 DEMO: {model} on {dataset}")
    print("="*80)
    
    # Training
    cmd = f"python support/models/train_extended.py {model} {dataset} --epochs {epochs} --device cpu"
    success = run_command(cmd, f"Training {model} on {dataset}")
    
    if success:
        # Defense testing
        for defense in ['dig', 'cig']:
            cmd = f"python attack_with_defense_extended.py {defense} {model} {dataset} --device cpu"
            run_command(cmd, f"Testing {defense} defense for {model} on {dataset}")

def check_datasets():
    """Check if datasets are available"""
    print("📁 Checking dataset availability...")
    print("="*60)
    
    datasets = {
        'IoTID20': 'support/dataset/IoTID20/train.csv',
        'WUSTL': 'support/dataset/WUSTL/wustl_iiot_2021_reduced.csv',
        'CICIoT2023': 'support/dataset/CICIoT2023/CIC_IoT_Dataset2023.csv'
    }
    
    for dataset, path in datasets.items():
        if os.path.exists(path):
            print(f"✅ {dataset}: {path}")
        else:
            print(f"❌ {dataset}: {path} (NOT FOUND)")

def main():
    parser = argparse.ArgumentParser(description='Demo extended pipeline')
    parser.add_argument('--mode', choices=['all', 'training', 'defense', 'specific'], 
                       default='all', help='Demo mode')
    parser.add_argument('--dataset', choices=['IoTID20', 'WUSTL', 'CICIoT2023'], 
                       help='Specific dataset for demo')
    parser.add_argument('--model', choices=['ResNetSEBlockIoT', 'SimpleCNNIoT', 'PureCNN', 'EfficientCNN'], 
                       help='Specific model for demo')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    
    args = parser.parse_args()
    
    print("🚀 BitShield Extended Pipeline Demo")
    print("="*80)
    
    # Check datasets first
    check_datasets()
    
    if args.mode == 'all':
        demo_training()
        demo_defense()
    elif args.mode == 'training':
        demo_training()
    elif args.mode == 'defense':
        demo_defense()
    elif args.mode == 'specific':
        if not args.dataset or not args.model:
            print("❌ Error: --dataset and --model are required for specific mode")
            return
        demo_specific(args.dataset, args.model, args.epochs)
    
    print("\n" + "="*80)
    print("🎉 Demo completed!")
    print("="*80)

if __name__ == '__main__':
    main()

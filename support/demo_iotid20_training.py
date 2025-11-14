#! /usr/bin/env python3

"""
Demo script showing how to use train.py with IoTID20 dataset
"""

import subprocess
import sys
import os

def run_training_example():
    """Run a simple training example with IoTID20 dataset"""
    
    print("=" * 60)
    print("IoTID20 Training Demo")
    print("=" * 60)
    
    # Available models for IoTID20
    models = [
        'ResNetSEBlockIoT',
        'SimpleCNNIoT', 
        'PureCNN',
        'EfficientCNN',
        'CustomModel',    # Alias for ResNetSEBlockIoT
        'CustomModel2'    # Alias for SimpleCNNIoT
    ]
    
    print("Available models for IoTID20:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    print("\nExample usage:")
    print("python support/models/train.py <model_name> IoTID20 --epochs 10 --batch-size 64")
    print("\nExample commands:")
    
    for model in models[:4]:  # Show first 4 models
        print(f"python support/models/train.py {model} IoTID20 --epochs 5 --batch-size 32")
    
    print("\n" + "=" * 60)
    print("Quick test with ResNetSEBlockIoT (1 epoch):")
    print("=" * 60)
    
    # Run a quick test
    try:
        cmd = [
            sys.executable, 
            "support/models/train.py", 
            "ResNetSEBlockIoT", 
            "IoTID20", 
            "--epochs", "1", 
            "--batch-size", "32"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Training completed successfully!")
            print("\nOutput:")
            print(result.stdout[-500:])  # Show last 500 characters
        else:
            print("✗ Training failed!")
            print("Error:")
            print(result.stderr[-500:])  # Show last 500 characters
            
    except subprocess.TimeoutExpired:
        print("Training timed out (this is normal for longer training)")
    except Exception as e:
        print(f"Error running training: {e}")

if __name__ == "__main__":
    run_training_example()

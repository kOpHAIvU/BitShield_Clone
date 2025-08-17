#!/usr/bin/env python3
"""
Simple script to run BitShield experiments without requiring TVM or model building.
This allows you to run basic experiments using pre-trained PyTorch models.
"""

import sys
import os
import argparse
import torch
import torchvision
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cfg
import dataman
from eval import evalutils

def run_basic_experiment(model_name, dataset, experiment_type="accuracy"):
    """
    Run basic experiments without requiring TVM compilation.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50', 'vgg16')
        dataset: Name of the dataset (e.g., 'CIFAR10', 'CIFAR100')
        experiment_type: Type of experiment ('accuracy', 'inference_time')
    """
    
    print(f"Running {experiment_type} experiment on {model_name} with {dataset}")
    
    # Get data loader
    try:
        data_loader = dataman.get_loader(dataset, 224, 'test', cfg.batch_size)
        print(f"Loaded {dataset} test dataset")
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return
    
    # Load pre-trained model
    try:
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
        elif model_name == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(pretrained=True)
        elif model_name == 'densenet121':
            model = torchvision.models.densenet121(pretrained=True)
        else:
            print(f"Model {model_name} not supported in basic mode")
            return
        
        model.eval()
        print(f"Loaded {model_name} model")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return
    
    # Run experiment
    if experiment_type == "accuracy":
        run_accuracy_test(model, data_loader, model_name, dataset)
    elif experiment_type == "inference_time":
        run_inference_time_test(model, data_loader, model_name, dataset)
    else:
        print(f"Experiment type {experiment_type} not supported")

def run_accuracy_test(model, data_loader, model_name, dataset):
    """Test model accuracy on the dataset."""
    print("Testing model accuracy...")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 10:  # Limit to first 10 batches for quick test
                break
                
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}: {correct}/{total} correct")
    
    accuracy = 100. * correct / total
    print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    
    # Save results
    results_dir = Path(cfg.project_root) / "results" / "basic_experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = results_dir / f"{model_name}_{dataset}_accuracy.txt"
    with open(result_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Correct: {correct}/{total}\n")
    
    print(f"Results saved to {result_file}")

def run_inference_time_test(model, data_loader, model_name, dataset):
    """Test model inference time."""
    print("Testing model inference time...")
    
    import time
    
    # Warm up
    for i, (data, _) in enumerate(data_loader):
        if i >= 2:
            break
        _ = model(data)
    
    # Test inference time
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= 10:  # Test 10 batches
                break
                
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            
            batch_time = (end_time - start_time) / data.size(0)  # Time per sample
            times.append(batch_time)
            
            if i % 2 == 0:
                print(f"Batch {i}: {batch_time*1000:.2f} ms per sample")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms per sample")
    
    # Save results
    results_dir = Path(cfg.project_root) / "results" / "basic_experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = results_dir / f"{model_name}_{dataset}_inference_time.txt"
    with open(result_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Average inference time: {avg_time*1000:.2f} ms per sample\n")
        f.write(f"Std inference time: {std_time*1000:.2f} ms per sample\n")
        f.write(f"Tested on {len(times)} batches\n")
    
    print(f"Results saved to {result_file}")

def main():
    parser = argparse.ArgumentParser(description='Run BitShield experiments without TVM')
    parser.add_argument('--model', '-m', type=str, default='resnet50',
                       help='Model name (resnet50, vgg16, mobilenet_v2, densenet121)')
    parser.add_argument('--dataset', '-d', type=str, default='CIFAR10',
                       help='Dataset name (CIFAR10, CIFAR100)')
    parser.add_argument('--experiment', '-e', type=str, default='accuracy',
                       choices=['accuracy', 'inference_time'],
                       help='Experiment type')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BitShield Basic Experiments (No TVM Required)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment}")
    print("=" * 60)
    
    run_basic_experiment(args.model, args.dataset, args.experiment)

if __name__ == "__main__":
    main()

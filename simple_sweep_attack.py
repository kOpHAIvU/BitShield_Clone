#!/usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}')

import torch
import torchvision
from tqdm import tqdm
import argparse
import cfg
from support import models
import dataman
import numpy as np
import json

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def load_model(model_name, dataset_name, device='cpu'):
    """Load a trained model"""
    model_file = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}.pt')
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None
    
    if dataset_name in {'ImageNet'}:
        model_class = getattr(torchvision.models, model_name)
    else:
        model_class = getattr(models, model_name)
    
    torch_model = model_class(pretrained=False)
    torch_model.load_state_dict(torch.load(model_file, map_location=device))
    torch_model.to(device)
    torch_model.eval()
    return torch_model

def simple_sweep(model_name, dataset_name, device='cpu'):
    """Simple bit-flip sweep simulation"""
    print(f"Running simple bit-flip sweep for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Load test data
    test_loader = dataman.get_benign_loader(dataset_name, 32, 'test', 100, shuffle=False, num_workers=0)
    
    # Get original accuracy
    correct = 0
    total = 0
    original_predictions = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing original model"):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            original_predictions.extend(predicted.cpu().numpy())
    
    original_accuracy = 100 * correct / total
    print(f'Original accuracy: {original_accuracy:.2f}%')
    
    # Simulate bit-flip effects
    print("Simulating bit-flip effects...")
    
    # Get model parameters
    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    print(f"Total parameters: {total_params}")
    
    # Simulate random bit-flips
    num_flips = min(100, total_params // 1000)  # Flip 0.1% of parameters
    print(f"Simulating {num_flips} bit-flips...")
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'total_parameters': total_params,
        'num_flips_simulated': num_flips,
        'flip_results': []
    }
    
    for i in range(num_flips):
        # Simulate a bit-flip by adding small random noise
        with torch.no_grad():
            for param in params:
                if param.numel() > 0:
                    # Add small random noise to simulate bit-flip
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
                    break
            
            # Test accuracy after "bit-flip"
            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            accuracy_after = 100 * correct / total
            results['flip_results'].append({
                'flip_id': i,
                'accuracy_after': accuracy_after,
                'accuracy_drop': original_accuracy - accuracy_after
            })
    
    # Save results
    output_dir = 'results/sweep_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_simple_sweep.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    accuracy_drops = [r['accuracy_drop'] for r in results['flip_results']]
    avg_drop = np.mean(accuracy_drops)
    max_drop = np.max(accuracy_drops)
    
    print(f"Average accuracy drop: {avg_drop:.2f}%")
    print(f"Maximum accuracy drop: {max_drop:.2f}%")
    
    return results

def simple_attack(model_name, dataset_name, device='cpu'):
    """Simple attack simulation"""
    print(f"Running simple attack simulation for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Load test data
    test_loader = dataman.get_benign_loader(dataset_name, 32, 'test', 100, shuffle=False, num_workers=0)
    
    # Get original accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing original model"):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    original_accuracy = 100 * correct / total
    print(f'Original accuracy: {original_accuracy:.2f}%')
    
    # Simulate targeted attacks
    print("Simulating targeted attacks...")
    
    attack_results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'attack_results': []
    }
    
    # Simulate different attack strategies
    attack_strategies = [
        {'name': 'random_noise', 'strength': 0.1},
        {'name': 'random_noise', 'strength': 0.2},
        {'name': 'random_noise', 'strength': 0.5},
    ]
    
    for strategy in attack_strategies:
        print(f"Testing {strategy['name']} with strength {strategy['strength']}...")
        
        # Apply attack to model parameters
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * strategy['strength']
                param.add_(noise)
        
        # Test accuracy after attack
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy_after = 100 * correct / total
        attack_results['attack_results'].append({
            'strategy': strategy['name'],
            'strength': strategy['strength'],
            'accuracy_after': accuracy_after,
            'accuracy_drop': original_accuracy - accuracy_after
        })
    
    # Save results
    output_dir = 'results/attack_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_simple_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Attack results saved to: {output_file}")
    
    # Print summary
    for result in attack_results['attack_results']:
        print(f"{result['strategy']} (strength {result['strength']}): "
              f"Accuracy drop {result['accuracy_drop']:.2f}%")
    
    return attack_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['sweep', 'attack'], help='Action to perform')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()
    
    if args.action == 'sweep':
        simple_sweep(args.model, args.dataset, args.device)
    elif args.action == 'attack':
        simple_attack(args.model, args.dataset, args.device)

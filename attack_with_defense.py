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
from support import torchdig

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

def attack_with_dig_protection(model_name, dataset_name, device='cpu'):
    """Attack simulation with DIG protection"""
    print(f"Running attack simulation with DIG protection for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Wrap model with DIG protection
    protected_model = torchdig.DIGProtectedModule(model)
    protected_model.to(device)
    protected_model.eval()
    
    # Load test data
    test_loader = dataman.get_benign_loader(dataset_name, 32, 'test', 100, shuffle=False, num_workers=0)
    train_loader = dataman.get_benign_loader(dataset_name, 32, 'train', 100, shuffle=False, num_workers=0)
    
    # Get original accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing original model"):
            x, y = x.to(device), y.to(device)
            y_pred = protected_model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    original_accuracy = 100 * correct / total
    print(f'Original accuracy: {original_accuracy:.2f}%')
    
    # Get suspicious score range for DIG
    print("Calculating suspicious score range...")
    sus_scores = []
    for x, y in tqdm(train_loader, desc="Calculating suspicious scores"):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)  # Enable gradients for input
        try:
            sus_score = protected_model.calc_sus_score(x).item()
            sus_scores.append(sus_score)
        except RuntimeError as e:
            print(f"Warning: Could not calculate suspicious score: {e}")
            sus_scores.append(0.0)  # Default value
        x.requires_grad_(False)  # Disable gradients
    
    sus_scores = np.array(sus_scores)
    sus_score_range = [np.percentile(sus_scores, 5), np.percentile(sus_scores, 95)]
    print(f'Suspicious score range: [{sus_score_range[0]:.2f}, {sus_score_range[1]:.2f}]')
    
    # Simulate attacks with DIG detection
    print("Simulating attacks with DIG protection...")
    
    attack_results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'sus_score_range': sus_score_range,
        'attack_results': []
    }
    
    # Different attack strengths
    attack_strengths = [0.1, 0.2, 0.5, 1.0]
    
    for strength in attack_strengths:
        print(f"Testing attack with strength {strength}...")
        
        # Apply attack to model parameters
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * strength
                param.add_(noise)
        
        # Test accuracy after attack
        correct = 0
        total = 0
        detected_attacks = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Check if DIG detects the attack
                x.requires_grad_(True)
                try:
                    sus_score = protected_model.calc_sus_score(x).item()
                    if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                        detected_attacks += 1
                        continue  # Skip this sample if detected
                except RuntimeError:
                    # If gradient calculation fails, assume not detected
                    pass
                x.requires_grad_(False)
                
                y_pred = protected_model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        if total > 0:
            accuracy_after = 100 * correct / total
        else:
            accuracy_after = 0.0
        
        detection_rate = 100 * detected_attacks / len(test_loader.dataset)
        
        attack_results['attack_results'].append({
            'strength': strength,
            'accuracy_after': accuracy_after,
            'accuracy_drop': original_accuracy - accuracy_after,
            'detection_rate': detection_rate,
            'samples_detected': detected_attacks,
            'samples_processed': total
        })
        
        print(f"  Accuracy after attack: {accuracy_after:.2f}%")
        print(f"  DIG detection rate: {detection_rate:.2f}%")
    
    # Save results
    output_dir = 'results/defense_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_dig_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n=== DIG Protection Summary ===")
    for result in attack_results['attack_results']:
        print(f"Strength {result['strength']}: "
              f"Accuracy drop {result['accuracy_drop']:.2f}%, "
              f"DIG detection {result['detection_rate']:.2f}%")
    
    return attack_results

def attack_with_cig_simulation(model_name, dataset_name, device='cpu'):
    """Simulate CIG protection (code integrity guard)"""
    print(f"Running attack simulation with CIG protection for {model_name} on {dataset_name}...")
    
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
    
    # Simulate CIG protection by checking parameter integrity
    print("Simulating CIG protection...")
    
    # Store original parameters for integrity check
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()
    
    attack_results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'attack_results': []
    }
    
    # Different attack strengths
    attack_strengths = [0.1, 0.2, 0.5, 1.0]
    
    for strength in attack_strengths:
        print(f"Testing attack with strength {strength}...")
        
        # Apply attack to model parameters
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * strength
                param.add_(noise)
        
        # Simulate CIG integrity check
        integrity_violations = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if name in original_params:
                # Check if parameter changed significantly (simulating CIG detection)
                diff = torch.abs(param.data - original_params[name])
                if torch.any(diff > 0.1):  # Threshold for CIG detection
                    integrity_violations += 1
                total_params += 1
        
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
        cig_detection_rate = 100 * integrity_violations / total_params if total_params > 0 else 0
        
        attack_results['attack_results'].append({
            'strength': strength,
            'accuracy_after': accuracy_after,
            'accuracy_drop': original_accuracy - accuracy_after,
            'cig_detection_rate': cig_detection_rate,
            'integrity_violations': integrity_violations,
            'total_params_checked': total_params
        })
        
        print(f"  Accuracy after attack: {accuracy_after:.2f}%")
        print(f"  CIG detection rate: {cig_detection_rate:.2f}%")
    
    # Save results
    output_dir = 'results/defense_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_cig_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n=== CIG Protection Summary ===")
    for result in attack_results['attack_results']:
        print(f"Strength {result['strength']}: "
              f"Accuracy drop {result['accuracy_drop']:.2f}%, "
              f"CIG detection {result['cig_detection_rate']:.2f}%")
    
    return attack_results

def attack_with_combined_protection(model_name, dataset_name, device='cpu'):
    """Attack simulation with both DIG and CIG protection"""
    print(f"Running attack simulation with combined DIG+CIG protection for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Wrap model with DIG protection
    protected_model = torchdig.DIGProtectedModule(model)
    protected_model.to(device)
    protected_model.eval()
    
    # Load test data
    test_loader = dataman.get_benign_loader(dataset_name, 32, 'test', 100, shuffle=False, num_workers=0)
    train_loader = dataman.get_benign_loader(dataset_name, 32, 'train', 100, shuffle=False, num_workers=0)
    
    # Get original accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing original model"):
            x, y = x.to(device), y.to(device)
            y_pred = protected_model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    original_accuracy = 100 * correct / total
    print(f'Original accuracy: {original_accuracy:.2f}%')
    
    # Get suspicious score range for DIG
    print("Calculating suspicious score range...")
    sus_scores = []
    for x, y in tqdm(train_loader, desc="Calculating suspicious scores"):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)  # Enable gradients for input
        try:
            sus_score = protected_model.calc_sus_score(x).item()
            sus_scores.append(sus_score)
        except RuntimeError as e:
            print(f"Warning: Could not calculate suspicious score: {e}")
            sus_scores.append(0.0)  # Default value
        x.requires_grad_(False)  # Disable gradients
    
    sus_scores = np.array(sus_scores)
    sus_score_range = [np.percentile(sus_scores, 5), np.percentile(sus_scores, 95)]
    print(f'Suspicious score range: [{sus_score_range[0]:.2f}, {sus_score_range[1]:.2f}]')
    
    # Store original parameters for CIG integrity check
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()
    
    # Simulate attacks with combined protection
    print("Simulating attacks with combined DIG+CIG protection...")
    
    attack_results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'sus_score_range': sus_score_range,
        'attack_results': []
    }
    
    # Different attack strengths
    attack_strengths = [0.1, 0.2, 0.5, 1.0]
    
    for strength in attack_strengths:
        print(f"Testing attack with strength {strength}...")
        
        # Apply attack to model parameters
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * strength
                param.add_(noise)
        
        # Check CIG integrity
        cig_violations = 0
        total_params = 0
        for name, param in model.named_parameters():
            if name in original_params:
                diff = torch.abs(param.data - original_params[name])
                if torch.any(diff > 0.1):
                    cig_violations += 1
                total_params += 1
        
        # Test accuracy and DIG detection
        correct = 0
        total = 0
        dig_detections = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Check DIG detection
                x.requires_grad_(True)
                try:
                    sus_score = protected_model.calc_sus_score(x).item()
                    if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                        dig_detections += 1
                        continue
                except RuntimeError:
                    # If gradient calculation fails, assume not detected
                    pass
                x.requires_grad_(False)
                
                y_pred = protected_model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        if total > 0:
            accuracy_after = 100 * correct / total
        else:
            accuracy_after = 0.0
        
        dig_detection_rate = 100 * dig_detections / len(test_loader.dataset)
        cig_detection_rate = 100 * cig_violations / total_params if total_params > 0 else 0
        combined_detection_rate = max(dig_detection_rate, cig_detection_rate)
        
        attack_results['attack_results'].append({
            'strength': strength,
            'accuracy_after': accuracy_after,
            'accuracy_drop': original_accuracy - accuracy_after,
            'dig_detection_rate': dig_detection_rate,
            'cig_detection_rate': cig_detection_rate,
            'combined_detection_rate': combined_detection_rate,
            'dig_detections': dig_detections,
            'cig_violations': cig_violations
        })
        
        print(f"  Accuracy after attack: {accuracy_after:.2f}%")
        print(f"  DIG detection rate: {dig_detection_rate:.2f}%")
        print(f"  CIG detection rate: {cig_detection_rate:.2f}%")
        print(f"  Combined detection rate: {combined_detection_rate:.2f}%")
    
    # Save results
    output_dir = 'results/defense_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_combined_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n=== Combined DIG+CIG Protection Summary ===")
    for result in attack_results['attack_results']:
        print(f"Strength {result['strength']}: "
              f"Accuracy drop {result['accuracy_drop']:.2f}%, "
              f"Combined detection {result['combined_detection_rate']:.2f}% "
              f"(DIG: {result['dig_detection_rate']:.1f}%, CIG: {result['cig_detection_rate']:.1f}%)")
    
    return attack_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('defense_type', choices=['dig', 'cig', 'combined'], help='Type of defense to test')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()
    
    if args.defense_type == 'dig':
        attack_with_dig_protection(args.model, args.dataset, args.device)
    elif args.defense_type == 'cig':
        attack_with_cig_simulation(args.model, args.dataset, args.device)
    elif args.defense_type == 'combined':
        attack_with_combined_protection(args.model, args.dataset, args.device)

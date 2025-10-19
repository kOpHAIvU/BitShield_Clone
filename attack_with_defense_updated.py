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
import numpy as np
import json
from support import torchdig
from support import torchdig_tabular

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
        torch_model = model_class(pretrained=False)
    else:
        model_class = getattr(models, model_name)
        if dataset_name == 'IoTID20':
            from support.dataman_iotid20 import preprocess_iotid20_data
            _, _, input_size, num_classes = preprocess_iotid20_data('support/dataset')
            torch_model = model_class(input_size=input_size, output_size=num_classes)
        else:
            torch_model = model_class(pretrained=False)
    
    torch_model.load_state_dict(torch.load(model_file, map_location='cpu'))
    torch_model.to(device)
    torch_model.eval()
    return torch_model

def attack_with_dig_protection(model_name, dataset_name, device='cpu'):
    """Attack simulation with DIG protection (uses Tabular DIG for IoTID20)"""
    print(f"Running attack simulation with DIG protection for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Load test data
    if dataset_name == 'IoTID20':
        from support.dataman_iotid20 import get_benign_loader_iotid20
        test_loader = get_benign_loader_iotid20('IoTID20', 32, 'test', batch_size=100)
        train_loader = get_benign_loader_iotid20('IoTID20', 32, 'train', batch_size=100)
        
        # Use Tabular DIG for IoTID20 (tabular data)
        print("Using Tabular DIG for IoTID20 dataset...")
        protected_model = torchdig_tabular.wrap_with_tabular_dig(model)
        protected_model.to(device)
        protected_model.eval()
        
        # Calculate suspicious score range for Tabular DIG
        from support.torchdig_tabular import calc_tabular_dig_range
        sus_score_range = calc_tabular_dig_range(protected_model, train_loader, device, n_batches=50)
        
    else:
        print(f"Dataset {dataset_name} not supported in this version")
        return
    
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
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Check if DIG detects the attack
            x.requires_grad_(True)
            try:
                sus_score = protected_model.calc_sus_score(x).item()
                if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                    detected_attacks += 1
                    # Still evaluate accuracy even if detected
            except RuntimeError as e:
                print(f"Warning: Could not calculate suspicious score: {e}")
            x.requires_grad_(False)
            
            # Evaluate accuracy
            with torch.no_grad():
                y_pred = protected_model(x)
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        accuracy_after = 100 * correct / total
        detection_rate = 100 * detected_attacks / (detected_attacks + total) if (detected_attacks + total) > 0 else 0
        
        attack_results['attack_results'].append({
            'strength': strength,
            'accuracy_after': accuracy_after,
            'accuracy_drop': original_accuracy - accuracy_after,
            'detection_rate': detection_rate
        })
        
        print(f"  Accuracy after attack: {accuracy_after:.2f}%")
        print(f"  DIG detection rate: {detection_rate:.2f}%")
        
        # Restore model for next test
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * strength
                param.sub_(noise)  # Reverse the attack
    
    # Save results
    output_dir = 'results/defense_results'
    ensure_dir_of(output_dir)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_dig_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DIG Protection Summary")
    print("="*60)
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
    if dataset_name == 'IoTID20':
        from support.dataman_iotid20 import get_benign_loader_iotid20
        test_loader = get_benign_loader_iotid20('IoTID20', 32, 'test', batch_size=100)
    else:
        print(f"Dataset {dataset_name} not supported in this version")
        return
    
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
    ensure_dir_of(output_dir)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_cig_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CIG Protection Summary")
    print("="*60)
    for result in attack_results['attack_results']:
        print(f"Strength {result['strength']}: "
              f"Accuracy drop {result['accuracy_drop']:.2f}%, "
              f"CIG detection {result['cig_detection_rate']:.2f}%")
    
    return attack_results

def attack_with_combined_protection(model_name, dataset_name, device='cpu'):
    """Attack simulation with combined DIG + CIG protection"""
    print(f"Running attack simulation with combined DIG + CIG protection for {model_name} on {dataset_name}...")
    
    # Run both DIG and CIG separately
    dig_results = attack_with_dig_protection(model_name, dataset_name, device)
    cig_results = attack_with_cig_simulation(model_name, dataset_name, device)
    
    # Combine results
    combined_results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': dig_results['original_accuracy'],
        'dig_results': dig_results,
        'cig_results': cig_results,
        'combined_analysis': []
    }
    
    # Analyze combined effectiveness
    for i, (dig_res, cig_res) in enumerate(zip(dig_results['attack_results'], cig_results['attack_results'])):
        strength = dig_res['strength']
        dig_detection = dig_res['detection_rate']
        cig_detection = cig_res['cig_detection_rate']
        
        # Combined detection rate (either DIG or CIG detects)
        combined_detection = max(dig_detection, cig_detection)
        
        combined_results['combined_analysis'].append({
            'strength': strength,
            'dig_detection': dig_detection,
            'cig_detection': cig_detection,
            'combined_detection': combined_detection,
            'improvement': combined_detection - max(dig_detection, cig_detection)
        })
    
    # Save combined results
    output_dir = 'results/defense_results'
    ensure_dir_of(output_dir)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_combined_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Combined results saved to: {output_file}")
    
    # Print combined summary
    print("\n" + "="*80)
    print("COMBINED DEFENSE SUMMARY")
    print("="*80)
    for result in combined_results['combined_analysis']:
        print(f"Strength {result['strength']}: "
              f"DIG {result['dig_detection']:.1f}% | "
              f"CIG {result['cig_detection']:.1f}% | "
              f"Combined {result['combined_detection']:.1f}%")
    print("="*80)
    
    return combined_results

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

#!/usr/bin/env python3
"""
Find effective attack parameters to achieve target accuracy degradation
"""

import sys
import os
import torch
import numpy as np
import copy
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)

import cfg
from support import models
from support.dataman_extended import get_benign_loader_extended, get_dataset_info
from support.models.quantized_layers import quan_Conv1d, quan_Linear, CustomBlock

# Import attack functions (copy from prepare_web_demo_models)
def int2bin(input_val, num_bits):
    """Convert signed integer → unsigned integer (2's complement)"""
    output = input_val if isinstance(input_val, torch.Tensor) else torch.tensor([input_val])
    output = output.clone()
    if num_bits == 1:
        output = output / 2 + 0.5
    elif num_bits > 1:
        mask = output.lt(0)
        output[mask] = 2**num_bits + output[mask]
    return output

def bin2int(input_val, num_bits):
    """Convert unsigned integer (2's complement) → signed integer"""
    input_tensor = input_val if isinstance(input_val, torch.Tensor) else torch.tensor([input_val])
    if num_bits == 1:
        output = input_tensor * 2 - 1
    elif num_bits > 1:
        mask = 2**(num_bits - 1) - 1
        output = -(input_tensor & ~mask) + (input_tensor & mask)
    return output

def _flip_one_bit_in_module_weight(module, elem_idx: int, bit_idx: int):
    """Flip one bit in module weight using quantized representation"""
    w = module.weight.data.view(-1)
    old_val = w[elem_idx].item()
    N_bits = getattr(module, 'N_bits', 8)
    bin_w = int2bin(torch.tensor([old_val]), N_bits).short().item()
    mask = 2 ** bit_idx
    bin_w_flipped = bin_w ^ mask
    new_val = bin2int(torch.tensor([bin_w_flipped]), N_bits).float().item()
    w[elem_idx] = new_val
    return (old_val, new_val - old_val, elem_idx, bit_idx)

def _get_quant_modules(model):
    """Get all quantized modules"""
    return [(n, m) for n, m in model.named_modules() if _is_quant_module(m)]

def _is_quant_module(m):
    """Check if module is quantized"""
    return isinstance(m, (quan_Conv1d, quan_Linear, CustomBlock))

def _random_flip_one_bit(model):
    """Randomly flip one bit in model"""
    quant_modules = _get_quant_modules(model)
    if not quant_modules:
        return None
    mod_name, mod = quant_modules[np.random.randint(len(quant_modules))]
    w_flat = mod.weight.data.view(-1)
    elem_idx = np.random.randint(w_flat.numel())
    N_bits = getattr(mod, 'N_bits', 8)
    bit_idx = np.random.randint(N_bits)
    result = _flip_one_bit_in_module_weight(mod, elem_idx, bit_idx)
    return {'module': mod_name, 'elem_idx': elem_idx, 'bit_idx': bit_idx, 'result': result}

def _compute_batch_loss(model, x, y, criterion):
    """Compute loss on batch"""
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        loss = criterion(outputs, y)
    return loss.item()

def _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16):
    """Progressive bit search attack"""
    quant_modules = _get_quant_modules(model)
    if not quant_modules:
        return None
    device = calib_x.device
    base_loss = _compute_batch_loss(model, calib_x, calib_y, criterion)
    best_delta = -1e9
    best_flip = None
    for _ in range(max_trials):
        mod_name, mod = quant_modules[np.random.randint(len(quant_modules))]
        w_flat = mod.weight.data.view(-1)
        if w_flat.numel() == 0:
            continue
        elem_idx = np.random.randint(w_flat.numel())
        N_bits = getattr(mod, 'N_bits', 8)
        bit_idx = np.random.randint(N_bits)
        old_w = w_flat[elem_idx].clone()
        result = _flip_one_bit_in_module_weight(mod, elem_idx, bit_idx)
        new_loss = _compute_batch_loss(model, calib_x, calib_y, criterion)
        delta = new_loss - base_loss
        w_flat[elem_idx] = old_w
        if delta > best_delta:
            best_delta = delta
            best_flip = {'module': mod_name, 'elem_idx': elem_idx, 'bit_idx': bit_idx, 'delta_loss': delta, 'result': result}
    if best_flip:
        mod_name = best_flip['module']
        for n, m in quant_modules:
            if n == mod_name:
                _flip_one_bit_in_module_weight(m, best_flip['elem_idx'], best_flip['bit_idx'])
                return best_flip
    return None

def find_effective_attack(model_name: str, dataset_name: str,
                         target_degradation: float = 0.4,  # Target 40% accuracy drop
                         attack_mode: str = 'pbs',
                         max_iterations: int = 200,
                         device: str = 'cuda'):
    """
    Find minimum attack iterations needed to achieve target degradation
    """
    print(f"\n{'='*80}")
    print(f"Finding Effective Attack Parameters")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Target Degradation: {target_degradation*100:.1f}%")
    print(f"Attack Mode: {attack_mode.upper()}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model, num_classes = load_model(model_name, dataset_name, device)
    
    # Get calibration data
    print("Loading calibration data...")
    calib_loader = get_benign_loader_extended(
        dataset_name, image_size=None, split='train',
        batch_size=128, shuffle=True, num_workers=0
    )
    calib_x, calib_y = next(iter(calib_loader))
    calib_x = calib_x.to(device)
    calib_y = calib_y.to(device)
    
    # Get test loader for evaluation
    test_loader = get_benign_loader_extended(
        dataset_name, image_size=None, split='test',
        batch_size=128, shuffle=False, num_workers=0
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate baseline
    print("Evaluating baseline...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    baseline_acc = correct / total
    print(f"Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n")
    
    target_acc = baseline_acc * (1 - target_degradation)
    print(f"Target Accuracy: {target_acc:.4f} ({target_acc*100:.2f}%)\n")
    
    # Binary search for effective iterations
    print("Searching for effective attack iterations...")
    print(f"{'Iterations':<12} {'Accuracy':<12} {'Degradation':<12} {'Status'}")
    print("-" * 60)
    
    best_iters = None
    min_iters = 1
    max_iters = max_iterations
    
    # Try different iteration counts
    test_points = [10, 25, 50, 75, 100, 150, 200]
    if max_iterations > 200:
        test_points.extend([250, 300, 400, 500])
    
    results = []
    
    for iters in test_points:
        if iters > max_iterations:
            break
            
        # Create fresh copy of model
        attacked_model = copy.deepcopy(model)
        attacked_model.eval()
        
        # Attack
        for i in range(iters):
            if attack_mode == 'pbs':
                _progressive_bit_search(attacked_model, criterion, calib_x, calib_y, max_trials=16)
            elif attack_mode == 'random':
                _random_flip_one_bit(attacked_model)
        
        # Evaluate
        attacked_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = attacked_model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = correct / total
        degradation = (baseline_acc - acc) / baseline_acc
        
        status = "✅ Target" if acc <= target_acc else "❌ Too high"
        print(f"{iters:<12} {acc:<12.4f} {degradation*100:<11.2f}% {status}")
        
        results.append((iters, acc, degradation))
        
        if acc <= target_acc and best_iters is None:
            best_iters = iters
            print(f"\n✅ Found effective attack: {iters} iterations")
            print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"   Degradation: {degradation*100:.2f}%")
            break
    
    if best_iters is None:
        # Try maximum iterations
        print(f"\nTrying maximum iterations ({max_iterations})...")
        attacked_model = copy.deepcopy(model)
        attacked_model.eval()
        
        for i in range(max_iterations):
            if attack_mode == 'pbs':
                _progressive_bit_search(attacked_model, criterion, calib_x, calib_y, max_trials=16)
            elif attack_mode == 'random':
                _random_flip_one_bit(attacked_model)
        
        attacked_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = attacked_model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = correct / total
        degradation = (baseline_acc - acc) / baseline_acc
        
        print(f"Max iterations ({max_iterations}): Accuracy = {acc:.4f} ({acc*100:.2f}%), Degradation = {degradation*100:.2f}%")
        
        if acc <= target_acc:
            best_iters = max_iterations
        else:
            print(f"\n⚠️  Warning: Even {max_iterations} iterations not enough!")
            print(f"   Consider using stronger attack mode or different approach")
            # Return the best we can do
            best_iters = max_iterations
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    print(f"Use --attack-mode {attack_mode} --attack-iters {best_iters}")
    print(f"\nExpected results:")
    print(f"  Original:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    if best_iters:
        # Find result for best_iters
        for iters, acc, deg in results:
            if iters == best_iters:
                print(f"  Attacked:  {acc:.4f} ({acc*100:.2f}%)")
                print(f"  Degradation: {deg*100:.2f}%")
                break
    
    return best_iters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find effective attack parameters')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('dataset_name', type=str, help='Dataset name')
    parser.add_argument('--target-degradation', type=float, default=0.4,
                       help='Target accuracy degradation (0.0-1.0)')
    parser.add_argument('--attack-mode', type=str, default='pbs', choices=['pbs', 'random'],
                       help='Attack mode')
    parser.add_argument('--max-iterations', type=int, default=200,
                       help='Maximum iterations to try')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    find_effective_attack(
        args.model_name,
        args.dataset_name,
        target_degradation=args.target_degradation,
        attack_mode=args.attack_mode,
        max_iterations=args.max_iterations,
        device=args.device
    )


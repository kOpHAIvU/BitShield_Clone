#!/usr/bin/env python3
"""
Prepare 3 models for web demo:
1. Original model (baseline)
2. Attacked model (after bit-flip attack)
3. Protected model (with OBFUS defense)

All models saved as .pt files for web integration.
"""

import sys
import os
import torch
import numpy as np
import copy
from tqdm import tqdm
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_root)

import cfg
from support import models
from support.dataman_extended import get_benign_loader_extended, get_dataset_info
from support.models.quantized_layers import quan_Conv1d, quan_Linear, CustomBlock
from support.obfus_sig import ObfusSigRuntime

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def load_model(model_name: str, dataset_name: str, device='cpu'):
    """Load a trained model"""
    model_file = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}.pt')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model_class = getattr(models, model_name)
    input_size, num_classes = get_dataset_info(dataset_name)
    torch_model = model_class(input_size=input_size, output_size=num_classes)
    
    torch_model.load_state_dict(torch.load(model_file, map_location='cpu'))
    torch_model.to(device)
    torch_model.eval()
    
    return torch_model, num_classes

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

def _is_quant_module(m):
    """Check if module is quantized"""
    return isinstance(m, (quan_Conv1d, quan_Linear, CustomBlock))

def _get_quant_modules(model):
    """Get all quantized modules"""
    return [(n, m) for n, m in model.named_modules() if _is_quant_module(m)]

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
    return {
        'module': mod_name,
        'elem_idx': elem_idx,
        'bit_idx': bit_idx,
        'result': result
    }

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
            best_flip = {
                'module': mod_name,
                'elem_idx': elem_idx,
                'bit_idx': bit_idx,
                'delta_loss': delta,
                'result': result
            }
    
    if best_flip:
        mod_name = best_flip['module']
        for n, m in quant_modules:
            if n == mod_name:
                _flip_one_bit_in_module_weight(m, best_flip['elem_idx'], best_flip['bit_idx'])
                return best_flip
    
    return None

def attack_model(model, criterion, calib_loader, attack_mode='pbs', attack_iters=25):
    """Attack model with specified attack mode"""
    print(f"\n{'='*60}")
    print(f"Attacking model with {attack_mode.upper()} ({attack_iters} iterations)...")
    print(f"{'='*60}")
    
    # Get calibration batch
    calib_x, calib_y = next(iter(calib_loader))
    device = next(model.parameters()).device
    calib_x = calib_x.to(device)
    calib_y = calib_y.to(device)
    
    attacked_model = copy.deepcopy(model)
    attacked_model.eval()
    
    for i in range(attack_iters):
        if attack_mode == 'pbs':
            result = _progressive_bit_search(attacked_model, criterion, calib_x, calib_y, max_trials=16)
        elif attack_mode == 'random':
            result = _random_flip_one_bit(attacked_model)
        else:
            raise ValueError(f"Unknown attack mode: {attack_mode}")
        
        if result:
            print(f"  Iteration {i+1}/{attack_iters}: Flipped bit in {result.get('module', 'unknown')}")
        else:
            print(f"  Iteration {i+1}/{attack_iters}: No flip applied")
    
    return attacked_model

def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = correct / total
    return accuracy

def prepare_web_demo_models(model_name: str, dataset_name: str, 
                            attack_mode: str = 'pbs', attack_iters: int = 25,
                            device: str = 'cuda', output_dir: str = None):
    """
    Prepare 3 models for web demo:
    1. Original model
    2. Attacked model
    3. Protected model (with OBFUS)
    """
    
    print(f"\n{'='*80}")
    print(f"Preparing Web Demo Models")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Attack Mode: {attack_mode}")
    print(f"Attack Iterations: {attack_iters}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Determine output directory
    if output_dir is None:
        output_dir = f'models/web_demo/{dataset_name}_{model_name}'
    ensure_dir_of(f'{output_dir}/original.pt')
    
    # Load original model
    print("Step 1: Loading original model...")
    original_model, num_classes = load_model(model_name, dataset_name, device)
    
    # Get data loaders
    print("Step 2: Loading datasets...")
    test_loader = get_benign_loader_extended(dataset_name, 32, 'test', batch_size=128, 
                                            shuffle=False, num_workers=0, image_size=None)
    calib_loader = get_benign_loader_extended(dataset_name, 32, 'train', batch_size=128, 
                                              shuffle=True, num_workers=0, image_size=None)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate original model
    print("\nStep 3: Evaluating original model...")
    original_acc = evaluate_model(original_model, test_loader, device, num_classes)
    print(f"Original Model Accuracy: {original_acc:.4f} ({original_acc*100:.2f}%)")
    
    # Save original model
    print("\nStep 4: Saving original model...")
    original_path = f'{output_dir}/original.pt'
    torch.save(original_model.state_dict(), original_path)
    print(f"✅ Saved: {original_path}")
    
    # Attack model
    print("\nStep 5: Attacking model...")
    attacked_model = attack_model(original_model, criterion, calib_loader, 
                                 attack_mode=attack_mode, attack_iters=attack_iters)
    
    # Evaluate attacked model
    print("\nStep 6: Evaluating attacked model...")
    attacked_acc = evaluate_model(attacked_model, test_loader, device, num_classes)
    print(f"Attacked Model Accuracy: {attacked_acc:.4f} ({attacked_acc*100:.2f}%)")
    
    if attacked_acc > original_acc * 0.8:
        print(f"⚠️  WARNING: Attack may not be effective (accuracy still high)")
        print(f"   Consider increasing attack_iters or using different attack mode")
    
    # Save attacked model
    print("\nStep 7: Saving attacked model...")
    attacked_path = f'{output_dir}/attacked.pt'
    torch.save(attacked_model.state_dict(), attacked_path)
    print(f"✅ Saved: {attacked_path}")
    
    # Create protected model (with OBFUS)
    print("\nStep 8: Creating protected model with OBFUS...")
    protected_model, _ = load_model(model_name, dataset_name, device)  # Fresh copy
    
    # Initialize OBFUS-SIG
    obfus_runtime = ObfusSigRuntime(
        model=protected_model,
        sig_period=500,
        sig_k=3.0,
        grad_norm_type='l1',
        normalize_grad=True,
        fp_threshold=0.1,
        fp_entropy_threshold=0.15,
        make_shadow=False,
        obfus_targets=['linear', 'conv1d'],
        max_obfus_layers=3,
        initial_reseed=False,  # Don't reseed immediately
        proactive_reseed_period=0,  # No proactive reseed
        allow_fallback=True,
        device=device
    )
    
    # Calibrate OBFUS
    print("  Calibrating OBFUS...")
    obfus_runtime.calibrate()
    
    # Get protected model (with OBFUS wrappers)
    protected_model = obfus_runtime.model
    
    # Evaluate protected model
    print("\nStep 9: Evaluating protected model...")
    protected_acc = evaluate_model(protected_model, test_loader, device, num_classes)
    print(f"Protected Model Accuracy: {protected_acc:.4f} ({protected_acc*100:.2f}%)")
    
    # Save protected model
    print("\nStep 10: Saving protected model...")
    protected_path = f'{output_dir}/protected.pt'
    
    # Save model state dict (OBFUS wrappers are part of the model)
    torch.save(protected_model.state_dict(), protected_path)
    print(f"✅ Saved: {protected_path}")
    
    # Save OBFUS runtime config for loading later
    obfus_config_path = f'{output_dir}/obfus_config.json'
    import json
    obfus_config = {
        'sig_period': 500,
        'sig_k': 3.0,
        'grad_norm_type': 'l1',
        'normalize_grad': True,
        'fp_threshold': 0.1,
        'fp_entropy_threshold': 0.15,
        'obfus_targets': ['linear', 'conv1d'],
        'max_obfus_layers': 3,
    }
    with open(obfus_config_path, 'w') as f:
        json.dump(obfus_config, f, indent=2)
    print(f"✅ Saved OBFUS config: {obfus_config_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Original Model:")
    print(f"  Path: {original_path}")
    print(f"  Accuracy: {original_acc:.4f} ({original_acc*100:.2f}%)")
    print(f"\nAttacked Model:")
    print(f"  Path: {attacked_path}")
    print(f"  Accuracy: {attacked_acc:.4f} ({attacked_acc*100:.2f}%)")
    print(f"  Degradation: {(original_acc - attacked_acc)*100:.2f}%")
    print(f"\nProtected Model:")
    print(f"  Path: {protected_path}")
    print(f"  Accuracy: {protected_acc:.4f} ({protected_acc*100:.2f}%)")
    print(f"  Protection: {(protected_acc - attacked_acc)*100:.2f}% recovery")
    print(f"\nOBFUS Config:")
    print(f"  Path: {obfus_config_path}")
    print(f"{'='*80}\n")
    
    # Verify requirements
    print("VERIFICATION:")
    print(f"✅ Original model can detect: {original_acc > 0.5}")
    print(f"❌ Attacked model cannot detect: {attacked_acc < original_acc * 0.5}")
    print(f"✅ Protected model can detect: {protected_acc > attacked_acc * 1.5}")
    
    if attacked_acc < original_acc * 0.5 and protected_acc > attacked_acc * 1.5:
        print("\n✅ All requirements met! Models ready for web demo.")
    else:
        print("\n⚠️  Warning: Some requirements not met. Consider adjusting attack parameters.")
    
    return {
        'original_path': original_path,
        'attacked_path': attacked_path,
        'protected_path': protected_path,
        'obfus_config_path': obfus_config_path,
        'original_acc': original_acc,
        'attacked_acc': attacked_acc,
        'protected_acc': protected_acc
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare models for web demo')
    parser.add_argument('model_name', type=str, help='Model name (e.g., ResNetSEBlockIoT)')
    parser.add_argument('dataset_name', type=str, help='Dataset name (e.g., IoTID20, CICIoT2023)')
    parser.add_argument('--attack-mode', type=str, default='pbs', choices=['pbs', 'random'],
                       help='Attack mode: pbs or random')
    parser.add_argument('--attack-iters', type=int, default=25,
                       help='Number of attack iterations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    prepare_web_demo_models(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        attack_mode=args.attack_mode,
        attack_iters=args.attack_iters,
        device=args.device,
        output_dir=args.output_dir
    )


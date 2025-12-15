#!/usr/bin/env python3
"""
Script to run comprehensive OBFUS experiments and collect metrics
Compares 3 stages:
1. Baseline (original model, no attack)
2. Attack without defense (4 attack modes)
3. Attack with OBFUS defense (4 attack modes)

Metrics: Accuracy, F1-Score, TPR (Recall), MCC
"""

import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Add project root to path (parent directory of obfus_visualization)
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_root)

import cfg
from support import models
from support.dataman_extended import get_benign_loader_extended, get_dataset_info
from support.models.quantized_layers import quan_Conv1d, quan_Linear, CustomBlock
from support.obfus_sig import ObfusSigRuntime

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def load_model(model_name: str, dataset_name: str, device='cpu'):
    """Load a trained model"""
    model_file = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}.pt')
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None
    
    model_class = getattr(models, model_name)
    input_size, num_classes = get_dataset_info(dataset_name)
    torch_model = model_class(input_size=input_size, output_size=num_classes)
    
    torch_model.load_state_dict(torch.load(model_file, map_location='cpu'))
    torch_model.to(device)
    torch_model.eval()
    
    return torch_model, num_classes

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculate ACC, F1, TPR, MCC with support for imbalanced datasets"""
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Balanced accuracy (better for imbalanced classes)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics (treat all classes equally)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    tpr_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted metrics (weight by class support - better for imbalanced)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    tpr_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    tpr_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))
    
    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'mcc': float(mcc),
        'tpr': float(tpr_macro),  # Macro-averaged
        'tpr_weighted': float(tpr_weighted),  # Weighted by support
        'f1': float(f1_macro),  # Macro-averaged
        'f1_weighted': float(f1_weighted),  # Weighted by support
        'precision': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'f1_per_class_std': float(np.std(f1_per_class)),  # Variance in F1
        'tpr_per_class_std': float(np.std(tpr_per_class)),  # Variance in TPR
        'class_distribution': class_distribution
    }

def evaluate_model(model, test_loader, device, num_classes: int) -> Dict[str, float]:
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)

def _is_quant_module(m):
    """Check if module is quantized"""
    return isinstance(m, (quan_Conv1d, quan_Linear, CustomBlock))

def _get_quant_modules(model):
    """Get all quantized modules"""
    return [(n, m) for n, m in model.named_modules() if _is_quant_module(m)]

def int2bin(input_val, num_bits):
    """Convert signed integer → unsigned integer (2's complement)"""
    output = input_val if isinstance(input_val, torch.Tensor) else torch.tensor([input_val])
    output = output.clone()
    
    if num_bits == 1:
        output = output / 2 + 0.5
    elif num_bits > 1:
        # Convert negative values to 2's complement
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
    """Flip one bit in module weight using quantized representation (like notebook)"""
    w = module.weight.data.view(-1)
    old_val = w[elem_idx].item()
    
    # Get quantization bitwidth
    N_bits = getattr(module, 'N_bits', 8)  # Default to 8 if not specified
    
    # Convert to quantized binary representation (2's complement)
    bin_w = int2bin(torch.tensor([old_val]), N_bits).short().item()
    
    # Create mask and flip the bit
    mask = 2 ** bit_idx
    bin_w_flipped = bin_w ^ mask
    
    # Convert back to float
    new_val = bin2int(torch.tensor([bin_w_flipped]), N_bits).float().item()
    
    w[elem_idx] = new_val
    return (old_val, new_val - old_val, elem_idx, bit_idx)

def _random_flip_one_bit(model):
    """Randomly flip one bit in model (using quantized representation)"""
    quant_modules = _get_quant_modules(model)
    if not quant_modules:
        return None
    
    # Random module
    mod_name, mod = quant_modules[np.random.randint(len(quant_modules))]
    w_flat = mod.weight.data.view(-1)
    elem_idx = np.random.randint(w_flat.numel())
    
    # ✅ Use N_bits instead of 32 (quantized representation)
    N_bits = getattr(mod, 'N_bits', 8)
    bit_idx = np.random.randint(N_bits)  # Flip within quantized bits
    
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
    
    trials = 0
    for _ in range(max_trials):
        if trials >= max_trials:
            break
        
        # Random module
        mod_name, mod = quant_modules[np.random.randint(len(quant_modules))]
        w_flat = mod.weight.data.view(-1)
        
        elem_idx = np.random.randint(w_flat.numel())
        
        # ✅ Use N_bits instead of 32 (quantized representation)
        N_bits = getattr(mod, 'N_bits', 8)
        bit_idx = np.random.randint(N_bits)
        
        # Flip bit
        old_w = w_flat[elem_idx].clone()
        result = _flip_one_bit_in_module_weight(mod, elem_idx, bit_idx)
        
        # Compute new loss
        new_loss = _compute_batch_loss(model, calib_x, calib_y, criterion)
        delta = new_loss - base_loss
        
        # Revert
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
        
        trials += 1
    
    # Apply best flip
    if best_flip:
        mod_name = best_flip['module']
        for n, m in quant_modules:
            if n == mod_name:
                _flip_one_bit_in_module_weight(m, best_flip['elem_idx'], best_flip['bit_idx'])
                break
    
    return best_flip

def attack_model(model, test_loader, device, num_classes: int, 
                 attack_mode: str = 'pbs', attack_iters: int = 25,
                 obfus_runtime=None) -> Dict[str, float]:
    """
    Attack model with bit-flips
    
    Args:
        model: Model to attack
        test_loader: Test data loader
        device: Device
        num_classes: Number of classes
        attack_mode: 'pbs', 'random', 'pbs2random', 'random2pbs'
        attack_iters: Number of attack iterations
        obfus_runtime: OBFUS runtime (if using defense)
    
    Returns:
        Metrics after attack
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Get calibration batch
    calib_x, calib_y = next(iter(test_loader))
    calib_x, calib_y = calib_x.to(device), calib_y.to(device)
    
    print(f"Attacking with mode: {attack_mode}, {attack_iters} iterations...")
    
    for i in range(attack_iters):
        # Determine attack type
        if attack_mode == 'pbs':
            flip_result = _progressive_bit_search(model, criterion, calib_x, calib_y)
        elif attack_mode == 'random':
            flip_result = _random_flip_one_bit(model)
        elif attack_mode == 'pbs2random':
            # Each iteration: PBS first, then RANDOM (like original code)
            _ = _progressive_bit_search(model, criterion, calib_x, calib_y)
            flip_result = _random_flip_one_bit(model)
        elif attack_mode == 'random2pbs':
            # Each iteration: RANDOM first, then PBS (like original code)
            _ = _random_flip_one_bit(model)
            flip_result = _progressive_bit_search(model, criterion, calib_x, calib_y)
        else:
            raise ValueError(f"Unknown attack mode: {attack_mode}")
        
        # OBFUS periodic check
        if obfus_runtime is not None:
            obfus_runtime.periodic_check(i)
    
    # Evaluate after attack
    return evaluate_model(model, test_loader, device, num_classes)

def run_experiments(model_name: str, dataset_name: str, device: str = 'cuda',
                   attack_modes: List[str] = None, attack_iters: int = 25,
                   obfus_config: Dict = None) -> Dict:
    """
    Run full experiment pipeline
    
    Returns:
        {
            'baseline': {metrics},
            'attack_no_defense': {
                'pbs': {metrics},
                'random': {metrics},
                ...
            },
            'attack_with_obfus': {
                'pbs': {metrics},
                'random': {metrics},
                ...
            }
        }
    """
    if attack_modes is None:
        attack_modes = ['pbs', 'random', 'pbs2random', 'random2pbs']
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'attack_iters': attack_iters,
        'baseline': {},
        'attack_no_defense': {},
        'attack_with_obfus': {}
    }
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*60}")
    
    test_loader = get_benign_loader_extended(dataset_name, None, 'test', batch_size=256)
    
    # Stage 1: Baseline (original model, no attack)
    print(f"\n{'='*60}")
    print("STAGE 1: Baseline Evaluation")
    print(f"{'='*60}")
    
    model, num_classes = load_model(model_name, dataset_name, device)
    baseline_metrics = evaluate_model(model, test_loader, device, num_classes)
    results['baseline'] = baseline_metrics
    
    print(f"Baseline Metrics:")
    print(f"  Accuracy:          {baseline_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {baseline_metrics['balanced_accuracy']:.4f}")
    print(f"  F1-Score (macro):  {baseline_metrics['f1']:.4f}")
    print(f"  F1-Score (weighted): {baseline_metrics['f1_weighted']:.4f}")
    print(f"  TPR (macro):       {baseline_metrics['tpr']:.4f}")
    print(f"  TPR (weighted):    {baseline_metrics['tpr_weighted']:.4f}")
    print(f"  MCC:               {baseline_metrics['mcc']:.4f}")
    print(f"  F1 per-class std:  {baseline_metrics['f1_per_class_std']:.4f}")
    print(f"  Class distribution: {baseline_metrics['class_distribution']}")
    
    # Stage 2: Attack without defense
    print(f"\n{'='*60}")
    print("STAGE 2: Attack Without Defense")
    print(f"{'='*60}")
    
    for attack_mode in attack_modes:
        print(f"\n--- Attack Mode: {attack_mode.upper()} ---")
        model, num_classes = load_model(model_name, dataset_name, device)
        attack_metrics = attack_model(model, test_loader, device, num_classes,
                                     attack_mode=attack_mode, attack_iters=attack_iters)
        results['attack_no_defense'][attack_mode] = attack_metrics
        
        print(f"After Attack Metrics:")
        print(f"  Accuracy:          {attack_metrics['accuracy']:.4f} (Δ: {attack_metrics['accuracy'] - baseline_metrics['accuracy']:.4f})")
        print(f"  Balanced Accuracy: {attack_metrics['balanced_accuracy']:.4f} (Δ: {attack_metrics['balanced_accuracy'] - baseline_metrics['balanced_accuracy']:.4f})")
        print(f"  F1-Score (macro):  {attack_metrics['f1']:.4f} (Δ: {attack_metrics['f1'] - baseline_metrics['f1']:.4f})")
        print(f"  F1-Score (weighted): {attack_metrics['f1_weighted']:.4f} (Δ: {attack_metrics['f1_weighted'] - baseline_metrics['f1_weighted']:.4f})")
        print(f"  TPR (macro):       {attack_metrics['tpr']:.4f} (Δ: {attack_metrics['tpr'] - baseline_metrics['tpr']:.4f})")
        print(f"  MCC:               {attack_metrics['mcc']:.4f} (Δ: {attack_metrics['mcc'] - baseline_metrics['mcc']:.4f})")
    
    # Stage 3: Attack with OBFUS defense
    if obfus_config:
        print(f"\n{'='*60}")
        print("STAGE 3: Attack With OBFUS Defense")
        print(f"{'='*60}")
        print(f"OBFUS Config: {obfus_config}")
        
        # Get probe loader for OBFUS calibration
        probe_loader = get_benign_loader_extended(dataset_name, None, 'train', batch_size=16)
        
        for attack_mode in attack_modes:
            print(f"\n--- Attack Mode: {attack_mode.upper()} with OBFUS ---")
            model, num_classes = load_model(model_name, dataset_name, device)
            
            # Initialize OBFUS
            obfus_runtime = ObfusSigRuntime(
                model=model,
                probe_loader=probe_loader,
                alert_mode=obfus_config.get('alert_mode', 'or'),
                sig_period=obfus_config.get('sig_period', 20),
                sig_k=obfus_config.get('sig_k', 3.0),
                grad_norm_type=obfus_config.get('grad_norm_type', 'l1'),
                normalize_grad=obfus_config.get('normalize_grad', False),
                fp_threshold=obfus_config.get('fp_threshold', 0.1),
                fp_entropy_threshold=obfus_config.get('fp_entropy_threshold', 0.15),
                make_shadow=obfus_config.get('make_shadow', False),
                obfus_targets=obfus_config.get('obfus_targets', ['linear', 'conv1d']),
                max_obfus_layers=obfus_config.get('max_obfus_layers', None),
                initial_reseed=obfus_config.get('initial_reseed', False),
                proactive_reseed_period=obfus_config.get('proactive_reseed_period', 10),
                allow_fallback=obfus_config.get('allow_fallback', True),
                device=device
            )
            
            # Calibrate
            print("Calibrating OBFUS...")
            obfus_runtime.calibrate()
            
            # ✅ CRITICAL: Evaluate baseline WITH OBFUS first!
            obfus_model = obfus_runtime.model
            print("Evaluating baseline WITH OBFUS (before attack)...")
            baseline_obfus_metrics = evaluate_model(obfus_model, test_loader, device, num_classes)
            print(f"Baseline WITH OBFUS:")
            print(f"  Accuracy:          {baseline_obfus_metrics['accuracy']:.4f} (Δ: {baseline_obfus_metrics['accuracy'] - baseline_metrics['accuracy']:.4f})")
            print(f"  Balanced Accuracy: {baseline_obfus_metrics['balanced_accuracy']:.4f} (Δ: {baseline_obfus_metrics['balanced_accuracy'] - baseline_metrics['balanced_accuracy']:.4f})")
            print(f"  F1-Score (macro):  {baseline_obfus_metrics['f1']:.4f} (Δ: {baseline_obfus_metrics['f1'] - baseline_metrics['f1']:.4f})")
            print(f"  F1-Score (weighted): {baseline_obfus_metrics['f1_weighted']:.4f} (Δ: {baseline_obfus_metrics['f1_weighted'] - baseline_metrics['f1_weighted']:.4f})")
            print(f"  MCC:               {baseline_obfus_metrics['mcc']:.4f} (Δ: {baseline_obfus_metrics['mcc'] - baseline_metrics['mcc']:.4f})")
            
            # Check if OBFUS broke the model
            if baseline_obfus_metrics['accuracy'] < 0.1:
                print("⚠️  WARNING: OBFUS destroyed model baseline accuracy! This suggests:")
                print("   - initial_reseed=True may be breaking the model")
                print("   - Try with initial_reseed=False or reduce max_obfus_layers")
            
            # Attack with OBFUS
            attack_metrics = attack_model(obfus_model, test_loader, device, num_classes,
                                         attack_mode=attack_mode, attack_iters=attack_iters,
                                         obfus_runtime=obfus_runtime)
            results['attack_with_obfus'][attack_mode] = attack_metrics
            results['attack_with_obfus'][attack_mode]['baseline_with_obfus'] = baseline_obfus_metrics
            
            print(f"\nAfter Attack (with OBFUS) Metrics:")
            print(f"  Accuracy:          {attack_metrics['accuracy']:.4f} (Δ: {attack_metrics['accuracy'] - baseline_obfus_metrics['accuracy']:.4f})")
            print(f"  Balanced Accuracy: {attack_metrics['balanced_accuracy']:.4f} (Δ: {attack_metrics['balanced_accuracy'] - baseline_obfus_metrics['balanced_accuracy']:.4f})")
            print(f"  F1-Score (macro):  {attack_metrics['f1']:.4f} (Δ: {attack_metrics['f1'] - baseline_obfus_metrics['f1']:.4f})")
            print(f"  F1-Score (weighted): {attack_metrics['f1_weighted']:.4f} (Δ: {attack_metrics['f1_weighted'] - baseline_obfus_metrics['f1_weighted']:.4f})")
            print(f"  MCC:               {attack_metrics['mcc']:.4f} (Δ: {attack_metrics['mcc'] - baseline_obfus_metrics['mcc']:.4f})")
            print(f"  F1 per-class std:  {attack_metrics['f1_per_class_std']:.4f}")
    
    return results

def save_results(results: Dict, output_file: str):
    """Save results to JSON"""
    ensure_dir_of(output_file)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run OBFUS experiments')
    parser.add_argument('model', type=str, help='Model name (e.g., ResNetSEBlockIoT)')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CICIoT2023, IoTID20)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--attack-iters', type=int, default=25, help='Number of attack iterations')
    parser.add_argument('--attack-modes', type=str, default='pbs,random,pbs2random,random2pbs',
                       help='Comma-separated attack modes')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    
    # OBFUS options
    parser.add_argument('--with-obfus', action='store_true', help='Enable OBFUS defense')
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
    
    attack_modes = args.attack_modes.split(',')
    
    obfus_config = None
    if args.with_obfus:
        obfus_config = {
            'alert_mode': 'or',
            'sig_period': args.sig_period,
            'sig_k': args.sig_k,
            'grad_norm_type': 'l1',
            'normalize_grad': False,
            'fp_threshold': 0.1,
            'fp_entropy_threshold': 0.15,
            'make_shadow': False,
            'obfus_targets': args.obfus_targets.split(','),
            'max_obfus_layers': args.obfus_max_layers,
            'initial_reseed': args.obfus_initial_reseed,
            'proactive_reseed_period': args.obfus_auto_reseed,
            'allow_fallback': True
        }
    
    results = run_experiments(
        model_name=args.model,
        dataset_name=args.dataset,
        device=args.device,
        attack_modes=attack_modes,
        attack_iters=args.attack_iters,
        obfus_config=obfus_config
    )
    
    # Save results
    if args.output is None:
        args.output = f'results/obfus_experiments/{args.dataset}_{args.model}_obfus_experiment.json'
    
    save_results(results, args.output)


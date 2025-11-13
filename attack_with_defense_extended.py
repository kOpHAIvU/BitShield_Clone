#!/usr/bin/env python3

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(project_root)

import torch
import torchvision
from tqdm import tqdm
import argparse
import cfg
from support import models
import numpy as np
import json
import csv
from support import torchdig
from support import torchdig_tabular
from support.dataman_extended import get_benign_loader_extended, get_dataset_info
from support.models.quantized_layers import quan_Conv1d, quan_Linear, CustomBlock

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
        # Get dataset info for model initialization
        try:
            input_size, num_classes = get_dataset_info(dataset_name)
            torch_model = model_class(input_size=input_size, output_size=num_classes)
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return None
    
    torch_model.load_state_dict(torch.load(model_file, map_location='cpu'))
    torch_model.to(device)
    torch_model.eval()
    return torch_model

def _is_quant_module(module):
    return isinstance(module, (quan_Conv1d, quan_Linear, CustomBlock))


def _flip_one_bit_in_module_weight(module, element_index=None, bit_index=None):
    """Flip exactly one bit in the given quantized module's weight tensor.
    Uses two's-complement flipping with module.N_bits and module.step_size.
    Returns (module_name, old_val, new_val, flat_idx, bit_idx).
    """
    if not hasattr(module, 'weight'):
        return None
    if not hasattr(module, 'N_bits'):
        return None
    if not hasattr(module, 'step_size'):
        return None

    weight = module.weight.data
    n_bits = int(getattr(module, 'N_bits', 8))
    step_size_tensor = getattr(module, 'step_size')
    step_size = step_size_tensor.detach().float().view(-1)[0].item() if isinstance(step_size_tensor, torch.Tensor) else float(step_size_tensor)
    if step_size == 0.0:
        # Fallback to a reasonable step size based on weight range
        with torch.no_grad():
            max_abs = weight.abs().max().item() if weight.numel() > 0 else 1.0
        step_size = max(1e-6, max_abs / max(1.0, (2 ** n_bits - 2) / 2.0))

    flat = weight.view(-1)
    numel = flat.numel()
    if numel == 0:
        return None
    if element_index is None:
        element_index = int(torch.randint(low=0, high=numel, size=(1,)).item())
    if bit_index is None:
        bit_index = int(torch.randint(low=0, high=n_bits, size=(1,)).item())

    # Quantize to integer domain
    with torch.no_grad():
        w_int = torch.round(flat / step_size).to(torch.int32)
        # Two's-complement to unsigned
        unsigned = w_int.clone()
        neg_mask = unsigned < 0
        unsigned[neg_mask] = (1 << n_bits) + unsigned[neg_mask]
        # Flip target bit
        toggle_mask = 1 << bit_index
        before_unsigned = unsigned[element_index].item()
        unsigned[element_index] = (unsigned[element_index].int() ^ toggle_mask)
        after_unsigned = unsigned[element_index].item()
        # Convert back to signed
        mask = (1 << (n_bits - 1)) - 1
        signed = -(unsigned & ~mask) + (unsigned & mask)
        # De-quantize back to float domain
        w_new = signed.to(flat.dtype) * step_size
        old_val = flat[element_index].item()
        new_val = w_new[element_index].item()
        flat.copy_(w_new)
    return (old_val, new_val, element_index, bit_index)


def _get_quant_modules(model):
    modules = []
    for name, module in model.named_modules():
        if _is_quant_module(module):
            modules.append((name, module))
    return modules


def _random_flip_one_bit(model):
    modules = _get_quant_modules(model)
    if not modules:
        return None
    name, module = modules[torch.randint(low=0, high=len(modules), size=(1,)).item()]
    result = _flip_one_bit_in_module_weight(module)
    return {'module': name, 'result': result}


def _compute_batch_loss(model, criterion, batch_x, batch_y):
    model.eval()
    with torch.no_grad():
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
    return float(loss.item())


def _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16):
    """Try flipping bits across random modules/elements; keep the flip that maximizes loss.
    Returns info dict.
    """
    modules = _get_quant_modules(model)
    if not modules:
        return None
    base_loss = _compute_batch_loss(model, criterion, calib_x, calib_y)
    best = {'delta': 0.0, 'apply': None, 'where': None}
    # Sample trials
    trials = min(max_trials, sum(m.weight.data.numel() > 0 for _, m in modules))
    for _ in range(trials):
        name, module = modules[torch.randint(low=0, high=len(modules), size=(1,)).item()]
        # Choose a random element/bit deterministically per trial
        weight = module.weight.data
        if weight.numel() == 0:
            continue
        elem_idx = int(torch.randint(low=0, high=weight.numel(), size=(1,)).item())
        bit_idx = int(torch.randint(low=0, high=int(getattr(module, 'N_bits', 8)), size=(1,)).item())
        # Save original
        old_val = weight.view(-1)[elem_idx].item()
        flip_info = _flip_one_bit_in_module_weight(module, elem_idx, bit_idx)
        trial_loss = _compute_batch_loss(model, criterion, calib_x, calib_y)
        delta = trial_loss - base_loss
        # Revert
        with torch.no_grad():
            weight.view(-1)[elem_idx] = torch.tensor(old_val, dtype=weight.dtype, device=weight.device)
        if delta > best['delta']:
            best = {'delta': delta, 'apply': (name, elem_idx, bit_idx), 'where': flip_info}
    # Apply best flip if improved
    if best['apply'] is not None and best['delta'] > 0:
        name, elem_idx, bit_idx = best['apply']
        # find module again
        for n, m in modules:
            if n == name:
                flip_result = _flip_one_bit_in_module_weight(m, elem_idx, bit_idx)
                return {
                    'module': name,
                    'elem_idx': elem_idx,
                    'bit_idx': bit_idx,
                    'delta_loss': best['delta'],
                    'result': flip_result,
                }
    # Otherwise apply a random flip as fallback
    return _random_flip_one_bit(model)


def _evaluate_with_dig(protected_model, test_loader, sus_score_range, device):
    correct = 0
    total = 0
    detected_attacks = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        x.requires_grad_(True)
        try:
            sus_score = protected_model.calc_sus_score(x).item()
            if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                # Count detections per-SAMPLE, but still evaluate accuracy (không skip)
                detected_attacks += batch_size
        except RuntimeError:
            pass
        x.requires_grad_(False)
        y_pred = protected_model(x)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy_after = 100 * correct / total if total > 0 else 0.0
    detection_rate = 100 * detected_attacks / len(test_loader.dataset)
    return accuracy_after, detection_rate, detected_attacks, total


def attack_with_dig_protection(model_name, dataset_name, device='cpu', attack_mode='noise', attack_iters=25):
    """Attack simulation with DIG protection (uses Tabular DIG for tabular datasets)"""
    print(f"Running attack simulation with DIG protection for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Load test data
    test_loader = get_benign_loader_extended(dataset_name, 32, 'test', batch_size=100)
    train_loader = get_benign_loader_extended(dataset_name, 32, 'train', batch_size=100)
    
    # Use Tabular DIG for tabular datasets
    if dataset_name in ['IoTID20', 'WUSTL', 'CICIoT2023']:
        print(f"Using Tabular DIG for {dataset_name} dataset...")
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
    
    if attack_mode == 'noise':
        # Different attack strengths (legacy noise attack)
        attack_strengths = [0.1, 0.2, 0.5, 1.0]
        for strength in attack_strengths:
            print(f"Testing attack with strength {strength}...")
            # Apply attack to model parameters
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.randn_like(param) * strength
                    param.add_(noise)
            # Evaluate after attack
            correct = 0
            total = 0
            detected_attacks = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)
                x.requires_grad_(True)
                try:
                    sus_score = protected_model.calc_sus_score(x).item()
                    if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                        # Đếm theo số mẫu, nhưng vẫn đánh giá accuracy
                        detected_attacks += batch_size
                except RuntimeError:
                    pass
                x.requires_grad_(False)
                y_pred = protected_model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy_after = 100 * correct / total if total > 0 else 0.0
            detection_rate = 100 * detected_attacks / len(test_loader.dataset)
            attack_results['attack_results'].append({
                'mode': 'noise',
                'strength': strength,
                'accuracy_after': accuracy_after,
                'accuracy_drop': original_accuracy - accuracy_after,
                'detection_rate': detection_rate
            })
            print(f"  Accuracy after attack: {accuracy_after:.2f}%")
            print(f"  DIG detection rate: {detection_rate:.2f}%")
    else:
        # Realistic bit-flip attacks from notebooks
        # Prepare a calibration batch for PBS selection
        calib_batch = next(iter(train_loader))
        calib_x, calib_y = calib_batch[0].to(device), calib_batch[1].to(device)
        criterion = torch.nn.CrossEntropyLoss()
        # Iterative attack with per-iteration logging
        iter_logs = []
        for i in range(int(attack_iters)):
            if attack_mode == 'pbs':
                info = _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16)
            elif attack_mode == 'random_flip':
                info = _random_flip_one_bit(model)
            elif attack_mode == 'pbs_to_random':
                _ = _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16)
                info = _random_flip_one_bit(model)
            elif attack_mode == 'random_to_pbs':
                _ = _random_flip_one_bit(model)
                info = _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16)
            else:
                info = _random_flip_one_bit(model)
            print(f"Iteration {i+1}/{attack_iters}: applied {attack_mode} step -> {info}")
            # Evaluate after each iteration
            acc_i, det_i, det_cnt_i, total_i = _evaluate_with_dig(protected_model, test_loader, sus_score_range, device)
            # Extract flip details if present
            module_name = info.get('module') if isinstance(info, dict) else None
            old_val = new_val = elem_idx = bit_idx = None
            if isinstance(info, dict):
                if 'result' in info and info['result'] is not None:
                    try:
                        old_val, new_val, elem_idx, bit_idx = info['result']
                    except Exception:
                        pass
                else:
                    elem_idx = info.get('elem_idx')
                    bit_idx = info.get('bit_idx')
            iter_logs.append([
                i + 1,
                attack_mode,
                module_name,
                old_val,
                new_val,
                elem_idx,
                bit_idx,
                f"{acc_i:.4f}",
                f"{det_i:.4f}",
                det_cnt_i,
                total_i,
            ])
        # Save per-iteration CSV
        output_dir = 'results/defense_results'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'{dataset_name}_{model_name}_{attack_mode}_iterlog.csv')
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration','mode','module','old_val','new_val','elem_idx','bit_idx',
                'accuracy_after_iter','dig_detection_rate_iter','samples_detected','samples_processed'
            ])
            writer.writerows(iter_logs)
        print(f"Per-iteration log saved to: {csv_path}")
        # Evaluate once after iterative attack
        
        correct = 0
        total = 0
        detected_attacks = 0
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            x.requires_grad_(True)
            try:
                sus_score = protected_model.calc_sus_score(x).item()
                if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                    # Count detections per-sample; do not skip accuracy
                    detected_attacks += batch_size
            except RuntimeError:
                # If gradient calculation fails, assume not detected
                pass
            x.requires_grad_(False)
            
            y_pred = protected_model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy_after = 100 * correct / total if total > 0 else 0.0
        detection_rate = 100 * detected_attacks / len(test_loader.dataset)
        attack_results['attack_results'].append({
            'mode': attack_mode,
            'iterations': int(attack_iters),
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
    ensure_dir_of(output_dir)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_dig_attack.json')
    
    with open(output_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n=== DIG Protection Summary ===")
    for result in attack_results['attack_results']:
        if 'strength' in result:
            header = f"Strength {result['strength']}"
        else:
            header = f"Mode {result.get('mode', 'unknown')} (iters {result.get('iterations', '?')})"
        print(f"{header}: Accuracy drop {result['accuracy_drop']:.2f}%, DIG detection {result['detection_rate']:.2f}%")
    
    return attack_results

def attack_with_cig_simulation(model_name, dataset_name, device='cpu', attack_mode='noise', attack_iters=25):
    """Simulate CIG protection (code integrity guard)"""
    print(f"Running attack simulation with CIG protection for {model_name} on {dataset_name}...")
    
    # Load model
    model = load_model(model_name, dataset_name, device)
    if model is None:
        return
    
    # Load data
    test_loader = get_benign_loader_extended(dataset_name, 32, 'test', batch_size=100)
    train_loader = get_benign_loader_extended(dataset_name, 32, 'train', batch_size=100)
    
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
    
    if attack_mode == 'noise':
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
                    diff = torch.abs(param.data - original_params[name])
                    if torch.any(diff > 0.1):
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
            accuracy_after = 100 * correct / total if total > 0 else 0.0
            cig_detection_rate = 100 * integrity_violations / total_params if total_params > 0 else 0
            attack_results['attack_results'].append({
                'mode': 'noise',
                'strength': strength,
                'accuracy_after': accuracy_after,
                'accuracy_drop': original_accuracy - accuracy_after,
                'cig_detection_rate': cig_detection_rate,
                'integrity_violations': integrity_violations,
                'total_params_checked': total_params
            })
            print(f"  Accuracy after attack: {accuracy_after:.2f}%")
            print(f"  CIG detection rate: {cig_detection_rate:.2f}%")
    else:
        # Bit-flip iterative attacks with per-iteration logging
        calib_batch = next(iter(train_loader))
        calib_x, calib_y = calib_batch[0].to(device), calib_batch[1].to(device)
        criterion = torch.nn.CrossEntropyLoss()
        iter_logs = []
        for i in range(int(attack_iters)):
            if attack_mode == 'pbs':
                info = _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16)
            elif attack_mode == 'random_flip':
                info = _random_flip_one_bit(model)
            elif attack_mode == 'pbs_to_random':
                _ = _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16)
                info = _random_flip_one_bit(model)
            elif attack_mode == 'random_to_pbs':
                _ = _random_flip_one_bit(model)
                info = _progressive_bit_search(model, criterion, calib_x, calib_y, max_trials=16)
            else:
                info = _random_flip_one_bit(model)
            print(f"Iteration {i+1}/{attack_iters}: applied {attack_mode} step -> {info}")
            # Integrity check after iteration
            integrity_violations = 0
            total_params = 0
            for name, param in model.named_parameters():
                if name in original_params:
                    diff = torch.abs(param.data - original_params[name])
                    if torch.any(diff > 0.1):
                        integrity_violations += 1
                    total_params += 1
            cig_rate_i = 100 * integrity_violations / total_params if total_params > 0 else 0
            # Accuracy after iteration
            correct_i = 0
            total_i = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    total_i += y.size(0)
                    correct_i += (predicted == y).sum().item()
            acc_i = 100 * correct_i / total_i if total_i > 0 else 0.0
            # Extract flip details
            module_name = info.get('module') if isinstance(info, dict) else None
            old_val = new_val = elem_idx = bit_idx = None
            if isinstance(info, dict):
                if 'result' in info and info['result'] is not None:
                    try:
                        old_val, new_val, elem_idx, bit_idx = info['result']
                    except Exception:
                        pass
                else:
                    elem_idx = info.get('elem_idx')
                    bit_idx = info.get('bit_idx')
            iter_logs.append([
                i + 1,
                attack_mode,
                module_name,
                old_val,
                new_val,
                elem_idx,
                bit_idx,
                f"{acc_i:.4f}",
                f"{cig_rate_i:.4f}",
                integrity_violations,
                total_params,
            ])
        # Save per-iteration CSV (CIG-specific filename)
        output_dir = 'results/defense_results'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'{dataset_name}_{model_name}_{attack_mode}_cig_iterlog.csv')
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration','mode','module','old_val','new_val','elem_idx','bit_idx',
                'accuracy_after_iter','cig_detection_rate_iter','integrity_violations','total_params_checked'
            ])
            writer.writerows(iter_logs)
        print(f"Per-iteration CIG log saved to: {csv_path}")
        # Final evaluation after iterative attack
        integrity_violations = 0
        total_params = 0
        for name, param in model.named_parameters():
            if name in original_params:
                diff = torch.abs(param.data - original_params[name])
                if torch.any(diff > 0.1):
                    integrity_violations += 1
                total_params += 1
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy_after = 100 * correct / total if total > 0 else 0.0
        cig_detection_rate = 100 * integrity_violations / total_params if total_params > 0 else 0
        attack_results['attack_results'].append({
            'mode': attack_mode,
            'iterations': int(attack_iters),
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
    print("\n=== CIG Protection Summary ===")
    for result in attack_results['attack_results']:
        if 'strength' in result:
            header = f"Strength {result['strength']}"
        else:
            header = f"Mode {result.get('mode', 'unknown')} (iters {result.get('iterations', '?')})"
        print(f"{header}: Accuracy drop {result['accuracy_drop']:.2f}%, CIG detection {result['cig_detection_rate']:.2f}%")
    
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
    parser.add_argument('dataset', type=str, choices=['IoTID20', 'WUSTL', 'CICIoT2023'], help='Dataset name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--attack-mode', type=str, default='noise', choices=['noise', 'pbs', 'random_flip', 'pbs_to_random', 'random_to_pbs'], help='Attack mode to simulate')
    parser.add_argument('--attack-iters', type=int, default=25, help='Number of attack iterations for bit-flip modes')
    args = parser.parse_args()
    
    if args.defense_type == 'dig':
        attack_with_dig_protection(args.model, args.dataset, args.device, attack_mode=args.attack_mode, attack_iters=args.attack_iters)
    elif args.defense_type == 'cig':
        attack_with_cig_simulation(args.model, args.dataset, args.device, attack_mode=args.attack_mode, attack_iters=args.attack_iters)
    elif args.defense_type == 'combined':
        attack_with_combined_protection(args.model, args.dataset, args.device)

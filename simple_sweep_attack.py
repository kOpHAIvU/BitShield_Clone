#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import time
import logging
from datetime import datetime

# ----------------------------
# Logging utilities
# ----------------------------
def setup_logger(log_level="INFO", log_file=None):
    lvl = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("bitflip")
    logger.setLevel(lvl)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def write_event(event_path, payload: dict):
    """Append structured event as JSONL."""
    if not event_path:
        return
    os.makedirs(os.path.dirname(event_path), exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    with open(event_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def device_info(logger, device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        try:
            if ":" in device:
                idx = int(device.split(":")[1])
        except Exception:
            idx = 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory
        reserved = torch.cuda.memory_reserved(idx)
        allocated = torch.cuda.memory_allocated(idx)
        logger.info(f"CUDA device: {name} (index {idx})")
        logger.info(f"VRAM total={total/1e9:.2f} GB, reserved={reserved/1e9:.2f} GB, allocated={allocated/1e9:.2f} GB")
    else:
        logger.info("Using CPU (or CUDA not available).")

def count_loader_items(loader):
    try:
        return len(loader.dataset)
    except Exception:
        # Fallback: approximate by iterating once
        n = 0
        for batch in loader:
            bx = batch[0]
            n += bx.size(0)
        return n

# ----------------------------
# Model I/O
# ----------------------------
def load_model(model_name, dataset_name, device='cpu', logger=None, event_log=None):
    """Load a trained model"""
    model_file = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}.pt')
    if not os.path.exists(model_file):
        msg = f"Model file not found: {model_file}"
        print(msg)
        if logger: logger.error(msg)
        write_event(event_log, {"ev": "model_not_found", "path": model_file})
        return None

    if logger:
        logger.info(f"Loading model: name={model_name}, dataset={dataset_name}")
        logger.info(f"Checkpoint: {model_file}")

    if dataset_name in {'ImageNet'}:
        model_class = getattr(torchvision.models, model_name)
        torch_model = model_class(pretrained=False)
    else:
        model_class = getattr(models, model_name)
        if dataset_name == 'IoTID20':
            from support.dataman_iotid20 import preprocess_iotid20_data
            if logger: logger.info("Inferring input_size/num_classes via preprocess_iotid20_data()")
            _, _, input_size, num_classes = preprocess_iotid20_data('support/dataset')
            torch_model = model_class(input_size=input_size, output_size=num_classes)
            if logger: logger.info(f"inferred: input_size={input_size}, num_classes={num_classes}")
        else:
            torch_model = model_class(pretrained=False)

    sd = torch.load(model_file, map_location='cpu')
    torch_model.load_state_dict(sd)

    torch_model.to(device)
    torch_model.eval()

    # Log parameter count
    total_params = sum(p.numel() for p in torch_model.parameters())
    trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    if logger:
        logger.info(f"Model loaded. total_params={total_params:,}, trainable_params={trainable_params:,}")
    write_event(event_log, {"ev": "model_loaded", "total_params": total_params, "trainable_params": trainable_params})

    return torch_model

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate_accuracy(model, loader, device='cpu'):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    return 100.0 * correct / max(total, 1)

# ----------------------------
# Experiments
# ----------------------------
def simple_sweep(model_name, dataset_name, device='cpu', logger=None, event_log=None, log_every=10):
    logger = logger or logging.getLogger("bitflip")
    logger.info(f"Running simple bit-flip sweep for model={model_name} dataset={dataset_name} device={device}")
    device_info(logger, device)

    # Load model
    model = load_model(model_name, dataset_name, device, logger, event_log)
    if model is None:
        return

    # Load test data
    if dataset_name == 'IoTID20':
        from support.dataman_iotid20 import get_benign_loader_iotid20
        logger.info("Loading IoTID20 test loader (benign)…")
        test_loader = get_benign_loader_iotid20('IoTID20', 32, 'test', batch_size=100)
    else:
        logger.error(f"Dataset {dataset_name} not supported in this simple version")
        return

    try:
        n_items = count_loader_items(test_loader)
        logger.info(f"Test set size ≈ {n_items}")
    except Exception:
        logger.warning("Could not determine test set size.")

    # Original accuracy
    t0 = time.time()
    original_accuracy = evaluate_accuracy(model, test_loader, device)
    logger.info(f"Original accuracy: {original_accuracy:.2f}% (eval took {time.time()-t0:.2f}s)")
    write_event(event_log, {"ev": "original_accuracy", "acc": original_accuracy})

    # Bit-flip-like noise simulation (current behavior)
    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    logger.info(f"Total parameters: {total_params:,}")

    num_flips = min(100, total_params // 1000)  # keep behavior
    logger.info(f"Configured flips: {num_flips}")
    write_event(event_log, {"ev": "sweep_config", "num_flips": num_flips, "total_params": total_params})

    results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'total_parameters': total_params,
        'num_flips_simulated': num_flips,
        'flip_results': []
    }

    pbar = tqdm(range(num_flips), desc="Sweep (flip-by-noise)", unit="flip")
    for i in pbar:
        with torch.no_grad():
            flipped = False
            for param in params:
                if param.numel() > 0:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
                    flipped = True
                    break
            if not flipped:
                logger.warning("No parameter was modified in this flip iteration.")

        acc_after = evaluate_accuracy(model, test_loader, device)
        drop = original_accuracy - acc_after
        results['flip_results'].append({
            'flip_id': i,
            'accuracy_after': acc_after,
            'accuracy_drop': drop
        })

        if (i % log_every) == 0 or i == num_flips - 1:
            logger.info(f"[flip {i+1}/{num_flips}] acc_after={acc_after:.2f}% (drop {drop:.2f}%)")
        write_event(event_log, {"ev": "flip_result", "flip_id": i, "acc_after": acc_after, "drop": drop})

    # Save results
    output_dir = 'results/sweep_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_simple_sweep.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy_drops = [r['accuracy_drop'] for r in results['flip_results']]
    avg_drop = float(np.mean(accuracy_drops)) if accuracy_drops else 0.0
    max_drop = float(np.max(accuracy_drops)) if accuracy_drops else 0.0

    logger.info(f"Sweep done. Results saved to: {output_file}")
    logger.info(f"Average accuracy drop: {avg_drop:.2f}% | Max drop: {max_drop:.2f}%")
    write_event(event_log, {"ev": "sweep_done", "output_file": output_file, "avg_drop": avg_drop, "max_drop": max_drop})

    return results

def simple_attack(model_name, dataset_name, device='cpu', logger=None, event_log=None, log_every=1):
    logger = logger or logging.getLogger("bitflip")
    logger.info(f"Running simple attack simulation for model={model_name} dataset={dataset_name} device={device}")
    device_info(logger, device)

    # Load model
    model = load_model(model_name, dataset_name, device, logger, event_log)
    if model is None:
        return

    # Load test data
    if dataset_name == 'IoTID20':
        from support.dataman_iotid20 import get_benign_loader_iotid20
        logger.info("Loading IoTID20 test loader (benign)…")
        test_loader = get_benign_loader_iotid20('IoTID20', 32, 'test', batch_size=100)
    else:
        logger.error(f"Dataset {dataset_name} not supported in this simple version")
        return

    try:
        n_items = count_loader_items(test_loader)
        logger.info(f"Test set size ≈ {n_items}")
    except Exception:
        logger.warning("Could not determine test set size.")

    # Original accuracy
    t0 = time.time()
    original_accuracy = evaluate_accuracy(model, test_loader, device)
    logger.info(f"Original accuracy: {original_accuracy:.2f}% (eval took {time.time()-t0:.2f}s)")
    write_event(event_log, {"ev": "original_accuracy", "acc": original_accuracy})

    attack_results = {
        'model': model_name,
        'dataset': dataset_name,
        'original_accuracy': original_accuracy,
        'attack_results': []
    }

    attack_strategies = [
        {'name': 'random_noise', 'strength': 0.1},
        {'name': 'random_noise', 'strength': 0.2},
        {'name': 'random_noise', 'strength': 0.5},
    ]
    logger.info(f"{len(attack_strategies)} attack strategies will be tested (note: noise accumulates in current behavior).")
    write_event(event_log, {"ev": "attack_config", "strategies": attack_strategies})

    for idx, strategy in enumerate(attack_strategies, 1):
        logger.info(f"[{idx}/{len(attack_strategies)}] Applying strategy={strategy['name']} strength={strategy['strength']}")
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * strategy['strength'])

        t1 = time.time()
        acc_after = evaluate_accuracy(model, test_loader, device)
        t2 = time.time()
        drop = original_accuracy - acc_after
        attack_results['attack_results'].append({
            'strategy': strategy['name'],
            'strength': strategy['strength'],
            'accuracy_after': acc_after,
            'accuracy_drop': drop
        })

        logger.info(f"→ acc_after={acc_after:.2f}% (drop {drop:.2f}%) | eval {t2 - t1:.2f}s")
        write_event(event_log, {
            "ev": "attack_eval",
            "idx": idx,
            "strategy": strategy['name'],
            "strength": strategy['strength'],
            "acc_after": acc_after,
            "drop": drop
        })

    # Save results
    output_dir = 'results/attack_results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_{model_name}_simple_attack.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(attack_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Attack results saved to: {output_file}")
    for r in attack_results['attack_results']:
        logger.info(f"{r['strategy']} (strength {r['strength']}): Accuracy drop {r['accuracy_drop']:.2f}%")
    write_event(event_log, {"ev": "attack_done", "output_file": output_file})

    return attack_results

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['sweep', 'attack'], help='Action to perform')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--log-file', type=str, default='logs/run.log', help='Path to a log file')
    parser.add_argument('--event-log', type=str, default='logs/events.jsonl', help='Path to JSONL event log')
    parser.add_argument('--log-every', type=int, default=10, help='Log every N steps (flips)')
    args = parser.parse_args()

    logger = setup_logger(args.log_level, args.log_file)
    logger.info(f"=== START action={args.action} model={args.model} dataset={args.dataset} device={args.device} ===")

    try:
        if args.action == 'sweep':
            simple_sweep(args.model, args.dataset, args.device, logger=logger, event_log=args.event_log, log_every=args.log_every)
        elif args.action == 'attack':
            simple_attack(args.model, args.dataset, args.device, logger=logger, event_log=args.event_log, log_every=args.log_every)
    except torch.cuda.OutOfMemoryError as e:
        logger.exception("CUDA OOM encountered. Consider reducing batch size or using --device cpu.")
        write_event(args.event_log, {"ev": "error", "type": "CUDA_OOM", "msg": str(e)})
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        write_event(args.event_log, {"ev": "interrupted"})
        sys.exit(130)
    except Exception as e:
        logger.exception("Unhandled exception.")
        write_event(args.event_log, {"ev": "error", "type": "unhandled", "msg": str(e)})
        sys.exit(2)
    finally:
        logger.info("=== END ===")

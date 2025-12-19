#!/usr/bin/env python3
"""
Helper script to load web demo models
Use this in your web application to load the 3 models
"""

import sys
import os
import torch
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_root)

import cfg
from support import models
from support.dataman_extended import get_dataset_info
from support.obfus_sig import ObfusSigRuntime

def load_web_demo_model(model_name: str, dataset_name: str, model_type: str, 
                       device='cpu', models_dir='models/web_demo'):
    """
    Load one of the 3 web demo models
    
    Args:
        model_name: Model name (e.g., 'ResNetSEBlockIoT')
        dataset_name: Dataset name (e.g., 'IoTID20')
        model_type: 'original', 'attacked', or 'protected'
        device: 'cpu' or 'cuda'
        models_dir: Base directory for models
    
    Returns:
        model: Loaded PyTorch model
        obfus_runtime: ObfusSigRuntime (only for 'protected' model, None otherwise)
    """
    
    # Construct model path
    model_dir = f'{models_dir}/{dataset_name}_{model_name}'
    model_path = f'{model_dir}/{model_type}.pt'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model architecture
    model_class = getattr(models, model_name)
    input_size, num_classes = get_dataset_info(dataset_name)
    model = model_class(input_size=input_size, output_size=num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    obfus_runtime = None
    
    # If protected model, also load OBFUS runtime
    if model_type == 'protected':
        obfus_config_path = f'{model_dir}/obfus_config.json'
        if os.path.exists(obfus_config_path):
            with open(obfus_config_path, 'r') as f:
                obfus_config = json.load(f)
            
            # Note: OBFUS runtime needs probe_loader for initialization
            # For web demo, you may need to provide a calibration loader
            # For now, we'll create a dummy loader or skip runtime initialization
            # The protected model can still be used without runtime for inference
            
            # Try to create a minimal probe loader (you may need to adjust this)
            try:
                from support.dataman_extended import get_benign_loader_extended
                # Create a small probe loader for calibration
                probe_loader = get_benign_loader_extended(
                    dataset_name, image_size=None, split='train', 
                    batch_size=32, shuffle=False, num_workers=0
                )
                
                # Initialize OBFUS runtime
                obfus_runtime = ObfusSigRuntime(
                    model=model,
                    probe_loader=probe_loader,  # Required!
                    alert_mode=obfus_config.get('alert_mode', 'or'),
                    sig_period=obfus_config.get('sig_period', 500),
                    sig_k=obfus_config.get('sig_k', 3.0),
                    grad_norm_type=obfus_config.get('grad_norm_type', 'l1'),
                    normalize_grad=obfus_config.get('normalize_grad', True),
                    fp_threshold=obfus_config.get('fp_threshold', 0.1),
                    fp_entropy_threshold=obfus_config.get('fp_entropy_threshold', 0.15),
                    make_shadow=obfus_config.get('make_shadow', False),
                    obfus_targets=obfus_config.get('obfus_targets', ['linear', 'conv1d']),
                    max_obfus_layers=obfus_config.get('max_obfus_layers', 3),
                    initial_reseed=False,
                    proactive_reseed_period=0,
                    device=device
                )
                
                # Calibrate (you may need calibration data for this)
                # obfus_runtime.calibrate()
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize OBFUS runtime: {e}")
                print("   Protected model loaded but OBFUS runtime not initialized")
                print("   Model can still be used for inference, but alerts won't work")
                obfus_runtime = None
        else:
            print(f"⚠️  Warning: OBFUS config not found at {obfus_config_path}")
            print("   Protected model loaded but OBFUS runtime not initialized")
    
    return model, obfus_runtime

def predict_with_model(model, x, obfus_runtime=None):
    """
    Make prediction with model (with optional OBFUS runtime)
    
    Args:
        model: PyTorch model
        x: Input tensor
        obfus_runtime: Optional OBFUS runtime (for protected model)
    
    Returns:
        predictions: Model predictions
        obfus_alerts: OBFUS alerts (if obfus_runtime provided)
    """
    model.eval()
    
    with torch.no_grad():
        if obfus_runtime is not None:
            # Use OBFUS runtime for protected model
            outputs = obfus_runtime.model(x)
            # Check for alerts (you may want to call periodic_check)
            obfus_alerts = None  # Implement alert checking if needed
        else:
            outputs = model(x)
            obfus_alerts = None
        
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted, obfus_alerts

# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load web demo models')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('dataset_name', type=str, help='Dataset name')
    parser.add_argument('--model-type', type=str, choices=['original', 'attacked', 'protected'],
                       default='original', help='Model type to load')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    
    args = parser.parse_args()
    
    print(f"Loading {args.model_type} model...")
    model, obfus_runtime = load_web_demo_model(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        device=args.device
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"   Type: {args.model_type}")
    print(f"   Device: {args.device}")
    if obfus_runtime:
        print(f"   OBFUS Runtime: Initialized")
    else:
        print(f"   OBFUS Runtime: None")


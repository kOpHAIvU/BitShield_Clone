#!/usr/bin/env python3

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import extended data manager
from support.dataman_extended import get_benign_loader_extended, get_dataset_info

# Import models
from support import models
import cfg

def calculate_class_weights(y_train, num_classes):
    """Calculate class weights to handle class imbalance"""
    if hasattr(y_train, 'cpu'):
        y_train = y_train.cpu().numpy()
    
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    return torch.FloatTensor(class_weights)

def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate comprehensive metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Calculate TPR and F1 for each class
    tpr_per_class = []
    f1_per_class = []
    
    for i in range(num_classes):
        if i in report:
            tpr_per_class.append(report[str(i)]['recall'])
            f1_per_class.append(report[str(i)]['f1-score'])
        else:
            tpr_per_class.append(0.0)
            f1_per_class.append(0.0)
    
    # Average TPR and F1
    avg_tpr = np.mean(tpr_per_class)
    avg_f1 = np.mean(f1_per_class)
    
    return {
        'Accuracy': accuracy,
        'MCC': mcc,
        'TPR': avg_tpr,
        'F1_Score': avg_f1,
        'Confusion_Matrix': cm.tolist(),
        'Per_Class_TPR': tpr_per_class,
        'Per_Class_F1': f1_per_class
    }

def train_model_extended(model_name, dataset_name, epochs=10, batch_size=256, device='cpu', 
                        use_class_weights=False, learning_rate=1e-3, weight_decay=1e-4):
    """Train model with extended dataset support"""
    
    print(f"Training {model_name} on {dataset_name} dataset...")
    print("="*60)
    
    # Get dataset info
    try:
        input_size, num_classes = get_dataset_info(dataset_name)
        print(f"Dataset: {dataset_name}")
        print(f"Input size: {input_size}")
        print(f"Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return None
    
    # Load model
    try:
        model_class = getattr(models, model_name)
        torch_model = model_class(input_size=input_size, output_size=num_classes)
        print(f"Model loaded: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    torch_model.to(device)
    
    # Load data
    try:
        train_loader = get_benign_loader_extended(dataset_name, 32, 'train', batch_size=batch_size)
        val_loader = get_benign_loader_extended(dataset_name, 32, 'val', batch_size=batch_size)
        test_loader = get_benign_loader_extended(dataset_name, 32, 'test', batch_size=batch_size)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        print("Calculating class weights to handle imbalance...")
        train_labels = []
        for _, y in train_loader:
            train_labels.extend(y.cpu().numpy())
        class_weights = calculate_class_weights(train_labels, num_classes)
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Setup loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer with improved settings
    optimizer = optim.Adam(torch_model.parameters(), 
                          lr=learning_rate, 
                          weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   mode='max', 
                                                   factor=0.5, 
                                                   patience=3, 
                                                   verbose=True)
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        torch_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_targets = []
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = torch_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        torch_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                x, y = x.to(device), y.to(device)
                outputs = torch_model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Calculate metrics
        val_metrics = calculate_metrics(val_targets, val_predictions, num_classes)
        
        # Update learning rate
        scheduler.step(val_metrics["Accuracy"])
        
        # Early stopping check
        current_val_acc = val_metrics["Accuracy"]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            patience_counter = 0
            # Save best model
            model_dir = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(torch_model.state_dict(), os.path.join(model_dir, f'{model_name}_best.pt'))
        else:
            patience_counter += 1
        
        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:3d}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_metrics["Accuracy"]*100:6.2f}% | '
              f'MCC: {val_metrics["MCC"]:6.3f}, TPR: {val_metrics["TPR"]:6.3f}, F1: {val_metrics["F1_Score"]:6.3f} | '
              f'LR: {current_lr:.2e} | Best: {best_val_acc*100:.2f}%')
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={early_stop_patience})")
            print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
            break
    
    # Load best model for final evaluation
    best_model_path = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}_best.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        torch_model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    torch_model.eval()
    test_correct = 0
    test_total = 0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            outputs = torch_model(x)
            _, predicted = torch.max(outputs.data, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    test_metrics = calculate_metrics(test_targets, test_predictions, num_classes)
    
    # Print final results
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test MCC: {test_metrics['MCC']:.3f}")
    print(f"Test TPR: {test_metrics['TPR']:.3f}")
    print(f"Test F1 Score: {test_metrics['F1_Score']:.3f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(test_metrics['Confusion_Matrix'])
    
    # Save final model
    model_dir = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(torch_model.state_dict(), os.path.join(model_dir, f'{model_name}.pt'))
    
    # Save results
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'test_accuracy': test_acc,
        'test_metrics': test_metrics,
        'best_val_accuracy': best_val_acc * 100,
        'training_epochs': epoch + 1,
        'input_size': input_size,
        'num_classes': num_classes
    }
    
    results_file = os.path.join(model_dir, f'{model_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to: {model_dir}")
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on extended datasets')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('dataset', type=str, choices=['IoTID20', 'WUSTL', 'CICIoT2023'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights to handle imbalance')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    
    args = parser.parse_args()
    
    train_model_extended(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_class_weights=args.use_class_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

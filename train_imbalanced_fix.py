#!/usr/bin/env python3

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(project_root)

from support.dataman_extended import preprocess_wustl_data, get_dataset_info
from support.models import ResNetSEBlockIoT, SimpleCNNIoT, PureCNN, EfficientCNN

def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced dataset"""
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return torch.FloatTensor(class_weights)

def train_with_imbalanced_fix(model_name, dataset_name, epochs=10, device='cpu', 
                             use_class_weights=True, use_focal_loss=True, 
                             use_sampling=True, learning_rate=0.001):
    """Train model with advanced imbalanced data handling"""
    
    print(f"Training {model_name} on {dataset_name} with imbalanced data fixes...")
    print("=" * 70)
    
    # Load data
    dataset_path = 'support/dataset'
    (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes = preprocess_wustl_data(dataset_path)
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    # Analyze class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass distribution in training set:")
    for class_id, count in zip(unique, counts):
        percentage = (count / len(y_train)) * 100
        print(f"  Class {class_id}: {count:,} samples ({percentage:.2f}%)")
    
    # Calculate class weights
    if use_class_weights:
        class_weights = calculate_class_weights(y_train)
        print(f"\nClass weights: {class_weights.numpy()}")
    else:
        class_weights = None
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Use weighted sampling for imbalanced data
    if use_sampling:
        # Calculate sample weights
        sample_weights = class_weights[y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    model_classes = {
        'ResNetSEBlockIoT': ResNetSEBlockIoT,
        'SimpleCNNIoT': SimpleCNNIoT,
        'PureCNN': PureCNN,
        'EfficientCNN': EfficientCNN
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_classes[model_name](input_size=input_size, output_size=num_classes)
    model = model.to(device)
    
    # Loss function
    if use_focal_loss:
        # Focal Loss for imbalanced data
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, class_weights=None):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.class_weights = class_weights
                
            def forward(self, inputs, targets):
                ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss
        
        criterion = FocalLoss(alpha=1, gamma=2, class_weights=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different learning rates for different layers
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'fc' in n or 'classifier' in n], 
         'lr': learning_rate * 10},  # Higher LR for final layers
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n and 'classifier' not in n], 
         'lr': learning_rate}  # Lower LR for feature extraction layers
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        # Calculate TPR and F1 for each class
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        avg_tpr = np.mean([report[f'{i}']['recall'] for i in range(num_classes) if f'{i}' in report])
        avg_f1 = np.mean([report[f'{i}']['f1-score'] for i in range(num_classes) if f'{i}' in report])
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{model_name}_{dataset_name}_best_imbalanced.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print(f'Epoch {epoch+1:2d}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:6.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:6.2f}% | '
              f'MCC: {mcc:.3f}, TPR: {avg_tpr:.3f}, F1: {avg_f1:.3f} | '
              f'LR: {current_lr:.2e} | Best: {best_val_acc:.2f}%')
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'{model_name}_{dataset_name}_best_imbalanced.pt'))
    model.eval()
    
    # Final evaluation on test set
    test_correct = 0
    test_total = 0
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(batch_y.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    test_mcc = matthews_corrcoef(all_test_labels, all_test_preds)
    
    # Detailed classification report
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {model_name} on {dataset_name}")
    print(f"{'='*70}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test MCC: {test_mcc:.3f}")
    
    # Per-class metrics
    report = classification_report(all_test_labels, all_test_preds, 
                                 target_names=[f'Class_{i}' for i in range(num_classes)],
                                 zero_division=0)
    print(f"\nDetailed Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return model, test_acc, test_mcc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with imbalanced data fixes')
    parser.add_argument('model', choices=['ResNetSEBlockIoT', 'SimpleCNNIoT', 'PureCNN', 'EfficientCNN'])
    parser.add_argument('dataset', default='WUSTL')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--no-focal-loss', action='store_true', help='Disable focal loss')
    parser.add_argument('--no-sampling', action='store_true', help='Disable weighted sampling')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Train model
    model, test_acc, test_mcc = train_with_imbalanced_fix(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        device=device,
        use_class_weights=not args.no_class_weights,
        use_focal_loss=not args.no_focal_loss,
        use_sampling=not args.no_sampling,
        learning_rate=args.learning_rate
    )
    
    print(f"\nTraining completed! Final Test Accuracy: {test_acc:.2f}%, MCC: {test_mcc:.3f}")

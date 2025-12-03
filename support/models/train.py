#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../..')

import torch
import torchvision
from tqdm import tqdm
import argparse
import cfg
from support import models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import IoTID20 data preprocessing
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataman_iotid20 import get_benign_loader_iotid20
    IOTID20_AVAILABLE = True
except ImportError:
    IOTID20_AVAILABLE = False
    print("Warning: IoTID20 data preprocessing not available")

# Import dataman only when needed
try:
    import dataman
    DATAMAN_AVAILABLE = True
except ImportError:
    DATAMAN_AVAILABLE = False
    print("Warning: dataman module not available")

def ensure_dir_of(filepath):  # No need to import utils
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def calculate_class_weights(y_train, num_classes):
    """
    Calculate class weights to handle class imbalance
    
    Args:
        y_train: Training labels
        num_classes: Number of classes
        
    Returns:
        torch.Tensor: Class weights for loss function
    """
    # Convert to numpy if needed
    if hasattr(y_train, 'cpu'):
        y_train = y_train.cpu().numpy()
    
    # Calculate balanced class weights
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    return torch.FloatTensor(class_weights)

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Calculate metrics for each class
    metrics = {}
    
    # Overall accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    metrics['Accuracy'] = accuracy
    
    # Per-class metrics
    tpr_list = []
    f1_list = []
    
    for i in range(num_classes):
        # True Positives, False Positives, False Negatives, True Negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # True Positive Rate (Sensitivity/Recall)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_list.append(tpr)
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
        f1_list.append(f1)
    
    # Average TPR and F1
    metrics['TPR'] = np.mean(tpr_list)
    metrics['F1_Score'] = np.mean(f1_list)
    
    # Matthews Correlation Coefficient (MCC)
    # For multi-class: MCC = (c*N - sum(tk*pk)) / sqrt((N^2 - sum(pk^2)) * (N^2 - sum(tk^2)))
    # where c = sum of diagonal elements, N = total samples, tk = true positives for class k, pk = predicted positives for class k
    
    N = np.sum(cm)
    c = np.trace(cm)
    
    # Sum of squares of true positives for each class
    tk_squared = np.sum(np.sum(cm, axis=1) ** 2)
    
    # Sum of squares of predicted positives for each class  
    pk_squared = np.sum(np.sum(cm, axis=0) ** 2)
    
    # Calculate MCC
    mcc_numerator = c * N - np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0))
    mcc_denominator = np.sqrt((N**2 - tk_squared) * (N**2 - pk_squared))
    
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    metrics['MCC'] = mcc
    metrics['Confusion_Matrix'] = cm
    
    return metrics

def plot_training_history(history, save_dir, model_name, dataset_name):
    """
    Plot training history including loss, accuracy, and other metrics
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Plot Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
    plt.title(f'Training and Validation Loss - {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f'{model_name}_loss_curves.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {loss_path}")
    
    # 2. Plot Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    plt.title(f'Training and Validation Accuracy - {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, f'{model_name}_accuracy_curves.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy curves saved to: {acc_path}")
    
    # 3. Plot Metrics (MCC, TPR, F1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, [m * 100 for m in history['val_mcc']], 'g-', label='MCC', linewidth=2, marker='o')
    plt.plot(epochs, [m * 100 for m in history['val_tpr']], 'm-', label='TPR (Recall)', linewidth=2, marker='s')
    plt.plot(epochs, [m * 100 for m in history['val_f1']], 'c-', label='F1 Score', linewidth=2, marker='^')
    plt.title(f'Validation Metrics - {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics_curves.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics curves saved to: {metrics_path}")
    
    # 4. Combined Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Convergence', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f'Model Convergence - {model_name} on {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(save_dir, f'{model_name}_convergence.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to: {combined_path}")

def plot_confusion_matrix(cm, num_classes, save_dir, model_name, dataset_name, class_names=None):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        num_classes: Number of classes
        save_dir: Directory to save plot
        model_name: Name of the model
        dataset_name: Name of the dataset
        class_names: Optional list of class names
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create labels for classes
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'}, linewidths=0.5)
    
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Also save raw confusion matrix (non-normalized)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    
    plt.title(f'Confusion Matrix (Raw Counts) - {model_name} on {dataset_name}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_raw_path = os.path.join(save_dir, f'{model_name}_confusion_matrix_raw.png')
    plt.savefig(cm_raw_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Raw confusion matrix saved to: {cm_raw_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", '-d', type=str, default='cpu')
    parser.add_argument('--output-root', type=str, default=cfg.models_dir)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights to handle imbalance')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument(
        '--balanced-sampler',
        action='store_true',
        help='Use a class-balanced sampler for IoTID20 training',
    )
    args = parser.parse_args()

    outfile = os.path.join(args.output_root, f'{args.dataset}/{args.model}/{args.model}.pt')
    if args.skip_existing and os.path.exists(outfile):
        print(f'Output file {outfile} exists, skipping')
        sys.exit(0)

    ensure_dir_of(outfile)

    # Set num_workers=0 to avoid multiprocessing issues in Docker
    if args.dataset == 'IoTID20' and IOTID20_AVAILABLE:
        print("Using IoTID20 data preprocessing...")
        train_loader = get_benign_loader_iotid20(
            args.dataset,
            args.image_size,
            'train',
            args.batch_size,
            shuffle=True,
            num_workers=0,
            use_balanced_sampler=args.balanced_sampler,
        )
        val_loader = get_benign_loader_iotid20(
            args.dataset,
            args.image_size,
            'test',
            args.batch_size,
            shuffle=True,
            num_workers=0,
            use_balanced_sampler=False,
        )
    elif DATAMAN_AVAILABLE:
        train_loader = dataman.get_benign_loader(args.dataset, args.image_size, 'train', args.batch_size, shuffle=True, num_workers=0)
        val_loader = dataman.get_benign_loader(args.dataset, args.image_size, 'test', args.batch_size, shuffle=True, num_workers=0)
    else:
        raise ImportError("Neither IoTID20 preprocessing nor dataman module is available")

    if args.dataset in {'ImageNet'}:  # Get from torchvision
        model_class = getattr(torchvision.models, args.model)
        torch_model = model_class(pretrained=False)
    elif args.dataset == 'IoTID20':
        # For IoTID20, we need to determine input size and number of classes
        # Get dataset info from the first batch
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[1]  # Number of features
        # Get the actual number of classes from the dataset
        all_labels = []
        for _, y in train_loader:
            all_labels.extend(y.tolist())
        num_classes = len(set(all_labels))
        
        print(f"IoTID20 dataset: {input_size} features, {num_classes} classes")
        
        # Map model names to actual classes
        model_mapping = {
            'ResNetSEBlockIoT': models.ResNetSEBlockIoT,
            'SimpleCNNIoT': models.SimpleCNNIoT,
            'PureCNN': models.PureCNN,
            'EfficientCNN': models.EfficientCNN,
            'CustomModel': models.ResNetSEBlockIoT,  # Backward compatibility
            'CustomModel2': models.SimpleCNNIoT,     # Backward compatibility
        }
        
        if args.model in model_mapping:
            model_class = model_mapping[args.model]
            torch_model = model_class(input_size=input_size, output_size=num_classes)
        else:
            raise ValueError(f"Unknown model for IoTID20: {args.model}. Available: {list(model_mapping.keys())}")
    else:
        model_class = getattr(models, args.model)
        torch_model = model_class(pretrained=False)
    torch_model.to(args.device)

    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights and args.dataset == 'IoTID20':
        print("Calculating class weights to handle imbalance...")
        # Get training labels for class weight calculation
        train_labels = []
        for _, y in train_loader:
            train_labels.extend(y.cpu().numpy())
        class_weights = calculate_class_weights(train_labels, num_classes)
        class_weights = class_weights.to(args.device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Setup loss function with class weights
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Setup optimizer with improved settings
    optimizer = torch.optim.Adam(torch_model.parameters(), 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='max', 
                                                         factor=0.5, 
                                                         patience=3)
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5
    
    # Training history for visualization
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_mcc': [],
        'val_tpr': [],
        'val_f1': []
    }

    for epoch in tqdm(range(args.epochs)):
        # Training phase
        torch_model.train(True)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            y_pred = torch_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            # Calculate training metrics
            train_loss += loss.item() * y.size(0)
            pred = y_pred.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        # Validation phase
        torch_model.train(False)
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = torch_model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * y.size(0)
                
                pred = y_pred.argmax(dim=1)
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        # Calculate comprehensive metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total * 100
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate detailed validation metrics
        val_metrics = calculate_metrics(val_targets, val_predictions, num_classes)
        
        # Store history for visualization
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics["Accuracy"] * 100)
        history['val_mcc'].append(val_metrics["MCC"])
        history['val_tpr'].append(val_metrics["TPR"])
        history['val_f1'].append(val_metrics["F1_Score"])
        
        # Update learning rate
        scheduler.step(val_metrics["Accuracy"])
        
        # Early stopping check
        current_val_acc = val_metrics["Accuracy"]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            patience_counter = 0
            # Save best model
            torch.save(torch_model.state_dict(), outfile.replace('.pt', '_best.pt'))
        else:
            patience_counter += 1
        
        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:3d}/{args.epochs}: '
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
    best_model_path = outfile.replace('.pt', '_best.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        torch_model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    
    # Get final predictions on validation set
    torch_model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(args.device), y.to(args.device)
            y_pred = torch_model(x)
            pred = y_pred.argmax(dim=1)
            final_predictions.extend(pred.cpu().numpy())
            final_targets.extend(y.cpu().numpy())
    
    # Calculate final metrics
    final_metrics = calculate_metrics(final_targets, final_predictions, num_classes)
    
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("-"*80)
    print("PERFORMANCE METRICS:")
    print(f"  Accuracy:     {final_metrics['Accuracy']*100:8.2f}%")
    print(f"  MCC:          {final_metrics['MCC']:8.3f}")
    print(f"  TPR (Recall): {final_metrics['TPR']:8.3f}")
    print(f"  F1 Score:     {final_metrics['F1_Score']:8.3f}")
    print("-"*80)
    
    # Confusion Matrix
    cm = final_metrics['Confusion_Matrix']
    print("CONFUSION MATRIX:")
    print("Predicted ->")
    print("Actual ↓")
    print("     ", end="")
    for i in range(num_classes):
        print(f"{i:8d}", end="")
    print()
    for i in range(num_classes):
        print(f"{i:3d}  ", end="")
        for j in range(num_classes):
            print(f"{cm[i,j]:8d}", end="")
        print()
    print("="*80)
    
    # Save model
    torch.save(torch_model.state_dict(), outfile)
    print(f'Model parameters saved to: {outfile}')
    
    # Generate and save visualization plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS...")
    print("="*80)
    
    # Get directory for saving plots
    plot_dir = os.path.dirname(outfile)
    
    # Plot training history
    plot_training_history(history, plot_dir, args.model, args.dataset)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, num_classes, plot_dir, args.model, args.dataset)
    
    print("\n" + "="*80)
    print("All visualization plots have been saved successfully!")
    print("="*80)

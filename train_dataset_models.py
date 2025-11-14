#!/usr/bin/env python3

"""
Script để train tất cả models trên một dataset cụ thể
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(project_root)

from support.models.train_extended import train_model_extended

def train_dataset_models(dataset_name, epochs=10, batch_size=256, device='cpu'):
    """Train tất cả models trên một dataset cụ thể"""
    
    # Định nghĩa các model
    models = [
        'ResNetSEBlockIoT',
        'SimpleCNNIoT', 
        'PureCNN',
        'EfficientCNN'
    ]
    
    print("="*60)
    print(f"TRAINING ALL MODELS ON {dataset_name}")
    print("="*60)
    print(f"Models: {models}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*60)
    
    results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 🚀 Training {model} on {dataset_name}")
        print("-" * 50)
        
        try:
            result = train_model_extended(
                model_name=model,
                dataset_name=dataset_name,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                use_class_weights=True,
                learning_rate=1e-3,
                weight_decay=1e-4
            )
            
            if result:
                results.append({
                    'model': model,
                    'dataset': dataset_name,
                    'accuracy': result['test_accuracy'],
                    'mcc': result['test_metrics']['MCC'],
                    'tpr': result['test_metrics']['TPR'],
                    'f1': result['test_metrics']['F1_Score'],
                    'best_val_acc': result['best_val_accuracy']
                })
                print(f"✅ {model}: {result['test_accuracy']:.2f}% acc, {result['test_metrics']['MCC']:.3f} MCC")
            else:
                print(f"❌ {model}: FAILED")
                
        except Exception as e:
            print(f"❌ {model}: ERROR - {str(e)}")
    
    # In kết quả tổng hợp
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {dataset_name}")
    print(f"{'='*60}")
    
    if results:
        # Sắp xếp theo accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"{'Rank':<5} {'Model':<18} {'Test Acc%':<10} {'Val Acc%':<10} {'MCC':<8} {'TPR':<8} {'F1':<8}")
        print("-" * 80)
        
        for i, r in enumerate(results, 1):
            print(f"{i:<5} {r['model']:<18} {r['accuracy']:<10.2f} {r['best_val_acc']:<10.2f} "
                  f"{r['mcc']:<8.3f} {r['tpr']:<8.3f} {r['f1']:<8.3f}")
        
        # Best model
        best = results[0]
        print(f"\n🏆 BEST MODEL: {best['model']}")
        print(f"   Test Accuracy: {best['accuracy']:.2f}%")
        print(f"   Validation Accuracy: {best['best_val_acc']:.2f}%")
        print(f"   MCC: {best['mcc']:.3f}")
        print(f"   TPR: {best['tpr']:.3f}")
        print(f"   F1 Score: {best['f1']:.3f}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train all models on a specific dataset')
    parser.add_argument('dataset', choices=['IoTID20', 'WUSTL', 'CICIoT2023'], 
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    try:
        results = train_dataset_models(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        if results:
            print(f"\n✅ Training completed successfully!")
            print(f"📊 Trained {len(results)} models on {args.dataset}")
        else:
            print(f"\n❌ No models were trained successfully")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main()


#!/usr/bin/env python3

"""
Script để train tất cả các model với tất cả datasets một cách tự động
"""

import sys
import os
import time
import json
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(project_root)

from support.models.train_extended import train_model_extended

def train_all_models():
    """Train tất cả các model với tất cả datasets"""
    
    # Định nghĩa các model và datasets
    models = [
        'ResNetSEBlockIoT',
        'SimpleCNNIoT', 
        'PureCNN',
        'EfficientCNN'
    ]
    
    datasets = [
        'IoTID20',
        'WUSTL', 
        'CICIoT2023'
    ]
    
    # Cấu hình training
    config = {
        'epochs': 10,
        'batch_size': 256,
        'device': 'cpu',  # Thay đổi thành 'cuda' nếu có GPU
        'use_class_weights': True,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4
    }
    
    # Lưu kết quả tổng hợp
    all_results = []
    start_time = datetime.now()
    
    print("="*80)
    print("TRAINING ALL MODELS ON ALL DATASETS")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Config: {config}")
    print("="*80)
    
    total_combinations = len(models) * len(datasets)
    current_combination = 0
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset}")
        print(f"{'='*60}")
        
        for model in models:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Training {model} on {dataset}")
            print("-" * 50)
            
            try:
                # Train model
                result = train_model_extended(
                    model_name=model,
                    dataset_name=dataset,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    device=config['device'],
                    use_class_weights=config['use_class_weights'],
                    learning_rate=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
                
                if result is not None:
                    # Thêm thông tin bổ sung
                    result['model'] = model
                    result['dataset'] = dataset
                    result['config'] = config
                    result['timestamp'] = datetime.now().isoformat()
                    
                    all_results.append(result)
                    
                    print(f"✓ SUCCESS: {model} on {dataset}")
                    print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
                    print(f"  Test MCC: {result['test_metrics']['MCC']:.3f}")
                    print(f"  Test TPR: {result['test_metrics']['TPR']:.3f}")
                    print(f"  Test F1: {result['test_metrics']['F1_Score']:.3f}")
                else:
                    print(f"✗ FAILED: {model} on {dataset}")
                    
            except Exception as e:
                print(f"✗ ERROR: {model} on {dataset} - {str(e)}")
                # Thêm thông tin lỗi vào kết quả
                error_result = {
                    'model': model,
                    'dataset': dataset,
                    'config': config,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'test_accuracy': 0.0,
                    'test_metrics': {'MCC': 0.0, 'TPR': 0.0, 'F1_Score': 0.0}
                }
                all_results.append(error_result)
    
    # Tính thời gian hoàn thành
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Total combinations: {total_combinations}")
    print(f"Successful trainings: {len([r for r in all_results if 'error' not in r])}")
    print(f"Failed trainings: {len([r for r in all_results if 'error' in r])}")
    
    # Lưu kết quả tổng hợp
    results_file = f"training_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_combinations': total_combinations,
                'successful': len([r for r in all_results if 'error' not in r]),
                'failed': len([r for r in all_results if 'error' in r])
            },
            'config': config,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # Tạo báo cáo tóm tắt
    create_summary_report(all_results, results_file.replace('.json', '_summary.txt'))
    
    return all_results

def create_summary_report(results, filename):
    """Tạo báo cáo tóm tắt kết quả training"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Thống kê tổng quan
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        f.write(f"Total combinations: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        # Kết quả theo dataset
        datasets = ['IoTID20', 'WUSTL', 'CICIoT2023']
        for dataset in datasets:
            f.write(f"DATASET: {dataset}\n")
            f.write("-" * 30 + "\n")
            
            dataset_results = [r for r in successful if r['dataset'] == dataset]
            if dataset_results:
                # Sắp xếp theo accuracy
                dataset_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
                
                f.write(f"{'Model':<20} {'Accuracy':<10} {'MCC':<8} {'TPR':<8} {'F1':<8}\n")
                f.write("-" * 60 + "\n")
                
                for result in dataset_results:
                    f.write(f"{result['model']:<20} "
                           f"{result['test_accuracy']:<10.2f} "
                           f"{result['test_metrics']['MCC']:<8.3f} "
                           f"{result['test_metrics']['TPR']:<8.3f} "
                           f"{result['test_metrics']['F1_Score']:<8.3f}\n")
            else:
                f.write("No successful results\n")
            
            f.write("\n")
        
        # Top performers
        f.write("TOP PERFORMERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Rank':<5} {'Model':<20} {'Dataset':<12} {'Accuracy':<10}\n")
        f.write("-" * 50 + "\n")
        
        all_successful = sorted(successful, key=lambda x: x['test_accuracy'], reverse=True)
        for i, result in enumerate(all_successful[:10], 1):
            f.write(f"{i:<5} {result['model']:<20} {result['dataset']:<12} {result['test_accuracy']:<10.2f}\n")
        
        f.write("\n")
        
        # Failed trainings
        if failed:
            f.write("FAILED TRAININGS\n")
            f.write("-" * 30 + "\n")
            for result in failed:
                f.write(f"{result['model']} on {result['dataset']}: {result['error']}\n")
    
    print(f"Summary report saved to: {filename}")

def main():
    """Main function"""
    try:
        results = train_all_models()
        print("\nTraining completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

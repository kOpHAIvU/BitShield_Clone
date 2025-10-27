#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

def analyze_wustl_dataset():
    """Analyze WUSTL dataset for class distribution and potential issues"""
    
    print("Analyzing WUSTL Dataset...")
    print("=" * 50)
    
    # Load data
    data_path = 'support/dataset/WUSTL/wustl_iiot_2021.csv'
    data = pd.read_csv(data_path, skipinitialspace=True)
    
    print(f"Total samples: {len(data)}")
    print(f"Total features: {len(data.columns)}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"WARNING: Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("OK: No missing values")
    
    # Remove duplicates
    data_clean = data.drop_duplicates()
    print(f"After removing duplicates: {len(data_clean)} samples")
    
    # Remove unnecessary columns
    columns_to_remove = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']
    available_columns = [col for col in columns_to_remove if col in data_clean.columns]
    data_clean = data_clean.drop(columns=available_columns)
    
    # Analyze target variable
    target_col = 'Traffic'
    if target_col in data_clean.columns:
        print(f"\nTarget variable analysis:")
        print(f"Unique values: {data_clean[target_col].nunique()}")
        
        # Class distribution
        class_counts = data_clean[target_col].value_counts()
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(data_clean)) * 100
            print(f"  {class_name}: {count:,} samples ({percentage:.2f}%)")
        
        # Check for extreme imbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 100:
            print("EXTREME CLASS IMBALANCE DETECTED!")
            print("   This will cause TPR=0 and F1=0 for minority classes")
        elif imbalance_ratio > 10:
            print("WARNING: Significant class imbalance detected")
        
        # Encode labels to see numeric distribution
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(data_clean[target_col])
        
        print(f"\nEncoded class distribution:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = label_encoder.inverse_transform([class_id])[0]
            percentage = (count / len(y_encoded)) * 100
            print(f"  Class {class_id} ({class_name}): {count:,} samples ({percentage:.2f}%)")
        
        # Check if any class has very few samples
        min_samples = counts.min()
        if min_samples < 100:
            print(f"ERROR: Some classes have very few samples (min: {min_samples})")
            print("   This will cause training issues")
        
        # Analyze feature distribution
        feature_columns = [col for col in data_clean.columns if col != target_col]
        print(f"\nFeature analysis:")
        print(f"Number of features: {len(feature_columns)}")
        
        # Check for constant features
        constant_features = []
        for col in feature_columns:
            if data_clean[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"WARNING: Constant features found: {len(constant_features)}")
            print(f"   {constant_features[:5]}...")  # Show first 5
        
        # Check for highly correlated features
        numeric_features = data_clean[feature_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            corr_matrix = data_clean[numeric_features].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                print(f"WARNING: Highly correlated features found: {len(high_corr_pairs)} pairs")
                print(f"   Examples: {high_corr_pairs[:3]}")
        
        # Test train/val split
        print(f"\nTesting train/val split:")
        try:
            X = data_clean[feature_columns].values
            y = y_encoded
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            print(f"  Train: {len(X_train)} samples")
            print(f"  Val: {len(X_val)} samples") 
            print(f"  Test: {len(X_test)} samples")
            
            # Check class distribution in splits
            print(f"\nClass distribution in splits:")
            for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
                unique, counts = np.unique(split_y, return_counts=True)
                print(f"  {split_name}:")
                for class_id, count in zip(unique, counts):
                    class_name = label_encoder.inverse_transform([class_id])[0]
                    percentage = (count / len(split_y)) * 100
                    print(f"    Class {class_id} ({class_name}): {count} samples ({percentage:.2f}%)")
            
        except Exception as e:
            print(f"ERROR: Error in train/val split: {e}")
    
    else:
        print(f"ERROR: Target column '{target_col}' not found!")
        print(f"Available columns: {list(data_clean.columns)}")

if __name__ == '__main__':
    analyze_wustl_dataset()

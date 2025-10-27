#!/usr/bin/env python3

import os
import sys
import glob

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class TabularDataset(Dataset):
    """Generic tabular dataset for PyTorch"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def preprocess_iotid20_data(dataset_path):
    """Preprocess IoTID20 dataset with enhanced preprocessing from dataman_iotid20.py"""
    print("Using IoTID20 data preprocessing...")
    
    # Check if train/test files exist, create them if not
    train_path = os.path.join(dataset_path, 'IoTID20', 'train.csv')
    test_path = os.path.join(dataset_path, 'IoTID20', 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Creating train/test splits from source data...")
        
        # Find source CSV files
        iotid20_dir = os.path.join(dataset_path, 'IoTID20')
        source_files = []
        
        # Look for source CSV files
        for file in os.listdir(iotid20_dir):
            if file.endswith('.csv') and file not in ['train.csv', 'test.csv']:
                source_files.append(os.path.join(iotid20_dir, file))
        
        if not source_files:
            raise FileNotFoundError(f'Could not find source CSV file in {iotid20_dir}')
        
        # Process source CSV
        print(f"Processing source CSV: {source_files[0]}")
        df = pd.read_csv(source_files[0], skipinitialspace=True)
        df = df.drop_duplicates()
        
        # Remove non-informative columns (from dataman_iotid20.py)
        columns_to_remove = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Fwd_PSH_Flags', 
                            'Fwd_URG_Flags', 'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg', 
                            'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg', 'Bwd_Pkts/b_Avg', 
                            'Bwd_Blk_Rate_Avg', 'Init_Fwd_Win_Byts', 'Fwd_Seg_Size_Min']
        
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Handle infinite and NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df = df.drop_duplicates()
        
        # Detect label column
        label_col_guess = 'Cat' if 'Cat' in df.columns else ('Label' if 'Label' in df.columns else 'label')
        labels = df[[label_col_guess]].copy()
        features = df.drop(columns=[c for c in ['Label', 'Cat', 'Sub_Cat'] if c in df.columns], errors='ignore')
        
        # Split data (80% train, 20% test)
        train_df, test_df = train_test_split(pd.concat([features, labels], axis=1), 
                                           test_size=0.2, random_state=100)
        train_df = train_df.rename(columns={label_col_guess: 'label'})
        test_df = test_df.rename(columns={label_col_guess: 'label'})
        
        # Save split data
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Created train.csv and test.csv at {iotid20_dir}")
    
    # Load the split data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Process training data first (like dataman_iotid20.py)
    train_feature_columns = [col for col in train_data.columns if col != 'label']
    X_train_raw = train_data[train_feature_columns].values
    y_train_raw = train_data['label'].values
    
    # Process test data
    test_feature_columns = [col for col in test_data.columns if col != 'label']
    X_test_raw = test_data[test_feature_columns].values
    y_test_raw = test_data['label'].values
    
    # Fit scaler and label_encoder on training data only (like dataman_iotid20.py)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    
    # Transform test data using fitted scaler and label_encoder
    X_test_scaled = scaler.transform(X_test_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)
    
    # Further split training into train/val (no stratify for IoTID20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, y_train_encoded, test_size=0.2, random_state=100
    )
    
    # Use test data as is
    X_test, y_test = X_test_scaled, y_test_encoded
    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train_encoded))
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes

def preprocess_wustl_data(dataset_path):
    """Preprocess WUSTL dataset according to notebook"""
    print("Using WUSTL data preprocessing...")
    
    # Load data
    data_path = os.path.join(dataset_path, 'WUSTL', 'wustl_iiot_2021.csv')
    data = pd.read_csv(data_path, skipinitialspace=True)
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Remove unnecessary columns (as in notebook)
    columns_to_remove = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']
    data.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # Separate labels and data (as in notebook)
    datalabel = data[['Traffic']]
    data = data.drop(columns=['Traffic'])
    
    # Initialize scaler and encoder
    scaler = StandardScaler()
    onc = LabelEncoder()
    
    # Split data first (as in notebook) - NO stratify
    X_train, X_test, y_train, y_test = train_test_split(data, datalabel, test_size=0.2, random_state=42)
    
    # Transform features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Encode labels (as in notebook)
    y_train = onc.fit_transform(y_train['Traffic'].to_numpy().ravel())
    y_test = onc.transform(y_test['Traffic'].to_numpy().ravel())
    
    # Further split training into train/val - NO stratify
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes

def preprocess_ciciot2023_data(dataset_path):
    """Preprocess CICIoT2023 dataset according to notebook"""
    print("Using CICIoT2023 data preprocessing...")
    
    # Load data
    data_path = os.path.join(dataset_path, 'CICIoT2023', 'CIC_IoT_Dataset2023.csv')
    data = pd.read_csv(data_path, skipinitialspace=True)
    
    # Remove duplicates and handle NaN values (as in notebook)
    data = data.drop_duplicates()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    
    # Separate labels and data (as in notebook)
    datalabel = data[['Label']]
    data = data.drop(columns=['Label'])
    
    # Normalize specific numerical columns (as in notebook)
    scaler = StandardScaler()
    data[['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate']] = scaler.fit_transform(
        data[['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate']]
    )
    
    # Label Encoding (as in notebook)
    onc = LabelEncoder()
    y = onc.fit_transform(datalabel['Label'].to_numpy().ravel())
    
    # Split data into training and testing sets (as in notebook) - random_state=100
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
    
    # Normalize X_train and X_test (as in notebook)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=100
    )
    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes

def get_benign_loader_extended(dataset_name, image_size, split, batch_size=100, shuffle=True, num_workers=0):
    """Get data loader for any supported dataset.
    For 'train' split, automatically apply WeightedRandomSampler when class imbalance is severe.
    """
    
    # Determine dataset path
    dataset_path = 'support/dataset'
    
    # Preprocess data based on dataset
    if dataset_name == 'IoTID20':
        (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes = preprocess_iotid20_data(dataset_path)
    elif dataset_name == 'WUSTL':
        (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes = preprocess_wustl_data(dataset_path)
    elif dataset_name == 'CICIoT2023':
        (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes = preprocess_ciciot2023_data(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Select data based on split
    if split == 'train':
        X, y = X_train, y_train
    elif split == 'val':
        X, y = X_val, y_val
    elif split == 'test':
        X, y = X_test, y_test
    else:
        raise ValueError(f"Unsupported split: {split}")
    
    # Create dataset and dataloader
    dataset = TabularDataset(X, y)

    # Use weighted sampling for imbalanced datasets on training split
    if split == 'train':
        # Compute class counts and imbalance ratio
        unique, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        min_count = counts.min() if counts.min() > 0 else 1
        imbalance_ratio = max_count / min_count

        if imbalance_ratio >= 50:  # enable when extremely imbalanced
            class_count = {cls: cnt for cls, cnt in zip(unique, counts)}
            class_weights = {cls: max_count / cnt for cls, cnt in class_count.items()}
            sample_weights = np.array([class_weights[int(label)] for label in y], dtype=np.float64)

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    
    return dataloader

def get_dataset_info(dataset_name):
    """Get dataset information (input_size, num_classes)"""
    dataset_path = 'support/dataset'
    
    if dataset_name == 'IoTID20':
        (_, _), (_, _), (_, _), input_size, num_classes = preprocess_iotid20_data(dataset_path)
    elif dataset_name == 'WUSTL':
        (_, _), (_, _), (_, _), input_size, num_classes = preprocess_wustl_data(dataset_path)
    elif dataset_name == 'CICIoT2023':
        (_, _), (_, _), (_, _), input_size, num_classes = preprocess_ciciot2023_data(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return input_size, num_classes

# Backward compatibility
def preprocess_iotid20_data_legacy(dataset_path):
    """Legacy function for backward compatibility"""
    return preprocess_iotid20_data(dataset_path)

def get_benign_loader_iotid20(dataset_name, image_size, split, batch_size=100, shuffle=True, num_workers=0):
    """Legacy function for backward compatibility"""
    return get_benign_loader_extended(dataset_name, image_size, split, batch_size, shuffle, num_workers)

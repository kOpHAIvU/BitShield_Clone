#!/usr/bin/env python3

import os
import sys

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
    """Preprocess IoTID20 dataset (existing implementation)"""
    print("Using IoTID20 data preprocessing...")
    
    # Load data
    train_path = os.path.join(dataset_path, 'IoTID20', 'train.csv')
    test_path = os.path.join(dataset_path, 'IoTID20', 'test.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Combine and preprocess
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Separate features and labels
    feature_columns = [col for col in data.columns if col != 'Label']
    X = data[feature_columns].values
    y = data['Label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    input_size = X_scaled.shape[1]
    num_classes = len(np.unique(y_encoded))
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes

def preprocess_wustl_data(dataset_path):
    """Preprocess WUSTL dataset"""
    print("Using WUSTL data preprocessing...")
    
    # Load data
    data_path = os.path.join(dataset_path, 'WUSTL', 'wustl_iiot_2021.csv')
    data = pd.read_csv(data_path, skipinitialspace=True)
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Remove unnecessary columns
    columns_to_remove = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']
    data = data.drop(columns=columns_to_remove)
    
    # Separate features and labels
    feature_columns = [col for col in data.columns if col != 'Traffic']
    X = data[feature_columns].values
    y = data['Traffic'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    input_size = X_scaled.shape[1]
    num_classes = len(np.unique(y_encoded))
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes

def preprocess_ciciot2023_data(dataset_path):
    """Preprocess CICIoT2023 dataset"""
    print("Using CICIoT2023 data preprocessing...")
    
    # Load data
    data_path = os.path.join(dataset_path, 'CICIoT2023', 'CIC_IoT_Dataset2023.csv')
    data = pd.read_csv(data_path, skipinitialspace=True)
    
    # Remove duplicates and handle NaN values
    data = data.drop(columns=['Cat'])
    data = data.dropna()
    
    # Separate features and labels
    feature_columns = [col for col in data.columns if col != 'Label']
    X = data[feature_columns].values
    y = data['Label'].values
    
    # Normalize specific numerical columns (as in original notebook)
    scaler = StandardScaler()
    numerical_cols = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate']
    numerical_indices = [data.columns.get_loc(col) for col in numerical_cols if col in data.columns]
    
    if numerical_indices:
        X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=100, stratify=y_encoded
    )
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=100, stratify=y_train
    )
    
    input_size = X_scaled.shape[1]
    num_classes = len(np.unique(y_encoded))
    
    print(f"Dataset info: {input_size} features, {num_classes} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size, num_classes

def get_benign_loader_extended(dataset_name, image_size, split, batch_size=100, shuffle=True, num_workers=0):
    """Get data loader for any supported dataset"""
    
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

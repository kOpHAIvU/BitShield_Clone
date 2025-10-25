#! /usr/bin/env python3

import os
import torch
import urllib.request
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class IoTID20Dataset(Dataset):
    """Dataset class for IoTID20 with preprocessing"""
    
    def __init__(self, csv_file, feature_cols=None, label_col='label', scaler=None, label_encoder=None):
        df = pd.read_csv(csv_file)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]
        self.feature_cols = feature_cols
        self.label_col = label_col
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values

        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        self.scaler = scaler

        if label_encoder is None:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        else:
            y = label_encoder.transform(y)
        self.label_encoder = label_encoder

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_iotid20_data(data_root, source_csv=None, download_url=None):
    """
    Preprocess IoTID20 dataset and create train/test splits
    
    Args:
        data_root (str): Root directory for data files
        source_csv (str, optional): Path to source CSV file
        download_url (str, optional): URL to download dataset
        
    Returns:
        tuple: (train_csv_path, test_csv_path, n_features, n_classes)
    """
    train_csv = os.path.join(data_root, 'train.csv')
    test_csv = os.path.join(data_root, 'test.csv')
    
    # Check if train/test files exist, create them if not
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("Creating train/test splits from source data...")
        
        # Find source CSV
        candidates = []
        if source_csv and os.path.isfile(source_csv):
            candidates = [source_csv]
        else:
            candidates = [p for p in glob.glob(os.path.join(data_root, '*.csv')) 
                         if os.path.basename(p).lower() not in {'train.csv', 'test.csv'}]
        
        if not candidates and download_url:
            print(f"Downloading data from {download_url}...")
            downloaded_path = os.path.join(data_root, 'IoT_Network_Intrusion_Dataset.csv')
            try:
                os.makedirs(data_root, exist_ok=True)
                urllib.request.urlretrieve(download_url, downloaded_path)
                candidates = [downloaded_path]
                print("Download completed!")
            except Exception as e:
                print(f"Download failed: {e}")
                return None
        
        if not candidates:
            raise FileNotFoundError(f'Could not find source CSV file at {data_root}')
        
        # Process source CSV
        print(f"Processing source CSV: {candidates[0]}")
        df = pd.read_csv(candidates[0], skipinitialspace=True)
        df = df.drop_duplicates()
        
        # Remove non-informative columns
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
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print(f"Created train.csv and test.csv at {data_root}")
    
    # Get dataset info
    train_df = pd.read_csv(train_csv, nrows=100)
    n_features = len([c for c in train_df.columns if c != 'label'])
    n_classes = train_df['label'].nunique() if 'label' in train_df.columns else train_df['Cat'].nunique()
    
    print(f"Dataset info: {n_features} features, {n_classes} classes")
    return train_csv, test_csv, n_features, n_classes


def get_iotid20_loader(csv_file, batch_size=256, shuffle=True, num_workers=0, 
                      feature_cols=None, label_col='label', scaler=None, label_encoder=None):
    """
    Create DataLoader for IoTID20 dataset
    
    Args:
        csv_file (str): Path to CSV file
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        feature_cols (list, optional): Feature columns
        label_col (str): Label column name
        scaler (StandardScaler, optional): Pre-fitted scaler
        label_encoder (LabelEncoder, optional): Pre-fitted label encoder
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = IoTID20Dataset(csv_file, feature_cols, label_col, scaler, label_encoder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), dataset


def get_benign_loader_iotid20(dataset_name, image_size, split, batch_size, shuffle=True, num_workers=0):
    """
    Compatibility function for train.py - creates IoTID20 loaders
    
    Args:
        dataset_name (str): Dataset name (should be 'IoTID20')
        image_size (int): Not used for IoTID20
        split (str): 'train' or 'test'
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle
        num_workers (int): Number of workers
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    if dataset_name != 'IoTID20':
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'IoTID20' for IoT intrusion detection.")
    
    # Preprocess data if needed
    data_root = 'support/dataset/IoTID20'
    train_csv, test_csv, n_features, n_classes = preprocess_iotid20_data(data_root)
    
    # Create loaders
    if split == 'train':
        loader, dataset = get_iotid20_loader(train_csv, batch_size, shuffle, num_workers)
    elif split == 'test':
        # Load train dataset first to get scaler and label_encoder
        _, train_dataset = get_iotid20_loader(train_csv, batch_size, False, num_workers)
        loader, _ = get_iotid20_loader(test_csv, batch_size, shuffle, num_workers, 
                                      scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    else:
        raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")
    
    return loader


import os
import urllib.request
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class IoTID20CSVDataset(Dataset):
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


def _standardize_and_split_source_csv(data_root, out_train, out_test, source_csv=None):
    # Try to find a source CSV inside data_root or use provided path
    candidates = []
    if source_csv and os.path.isfile(source_csv):
        candidates = [source_csv]
    else:
        # 1) CSVs in the given data_root
        candidates = [p for p in glob.glob(os.path.join(data_root, '*.csv')) if os.path.basename(p).lower() not in {'train.csv', 'test.csv'}]
        # 2) Known common locations (including the one used in BFA/main.py) if not found
        if not candidates:
            common_paths = [
                r"D:\Sukem\NCKH\Dataset\IoT_Network_Intrusion_Dataset\IoT_Network_Intrusion_Dataset.csv",
                r"D:\Datasets\IoT_Network_Intrusion_Dataset.csv",
                r"D:\Dataset\IoT_Network_Intrusion_Dataset.csv",
                os.path.join(os.path.dirname(data_root), 'IoT_Network_Intrusion_Dataset.csv'),
            ]
            candidates = [p for p in common_paths if os.path.isfile(p)]
    if not candidates:
        return False
    src = candidates[0]
    df = pd.read_csv(src, skipinitialspace=True)
    # Clean similar to BFA/main.py
    df = df.drop_duplicates()
    # Drop known non-feature columns if they exist
    for col in ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp',
                'Fwd_PSH_Flags', 'Fwd_URG_Flags', 'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg',
                'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg', 'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg',
                'Init_Fwd_Win_Byts', 'Fwd_Seg_Size_Min']:
        if col in df.columns:
            df = df.drop(columns=[col])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.drop_duplicates()
    # Prefer label from 'Cat' if present, else 'Label', else try 'label'
    label_col_guess = 'Cat' if 'Cat' in df.columns else ('Label' if 'Label' in df.columns else ('label' if 'label' in df.columns else None))
    if label_col_guess is None:
        # Fallback: last column
        label_col_guess = df.columns[-1]
    labels = df[[label_col_guess]].copy()
    features = df.drop(columns=[c for c in ['Label', 'Cat', 'Sub_Cat'] if c in df.columns], errors='ignore')
    # Standardize and encode inside dataset class later; here only split and rename label
    train_df, test_df = train_test_split(pd.concat([features, labels], axis=1), test_size=0.2, random_state=100)
    # Normalize label column name to 'label'
    train_df = train_df.rename(columns={label_col_guess: 'label'})
    test_df = test_df.rename(columns={label_col_guess: 'label'})
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    return True


def _download_csv(download_url: str, dest_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(download_url, dest_path)
        return True
    except Exception:
        return False


def build_iotid20_loaders(data_root, batch_size=256, num_workers=0, source_csv=None, download_url: str = None):
    train_csv = os.path.join(data_root, 'train.csv')
    test_csv = os.path.join(data_root, 'test.csv')
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        ok = _standardize_and_split_source_csv(data_root, train_csv, test_csv, source_csv=source_csv)
        if not ok and download_url:
            # Attempt to download a source CSV then split
            downloaded = _download_csv(download_url, os.path.join(data_root, 'IoT_Network_Intrusion_Dataset.csv'))
            if downloaded:
                ok = _standardize_and_split_source_csv(data_root, train_csv, test_csv, source_csv=os.path.join(data_root, 'IoT_Network_Intrusion_Dataset.csv'))
        if not ok:
            raise FileNotFoundError(f'Missing IoTID20 CSVs at {train_csv} and {test_csv}. Please place CSVs or a single source CSV in folder.')

    # Prefer standardized label name
    label_col = 'label' if 'label' in pd.read_csv(train_csv, nrows=1).columns else 'Cat'
    train_ds = IoTID20CSVDataset(train_csv, label_col=label_col)
    test_ds = IoTID20CSVDataset(test_csv, feature_cols=train_ds.feature_cols, label_col=label_col,
                                scaler=train_ds.scaler, label_encoder=train_ds.label_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_features = train_ds.X.shape[1]
    n_classes = len(train_ds.label_encoder.classes_)
    return train_loader, test_loader, n_features, n_classes


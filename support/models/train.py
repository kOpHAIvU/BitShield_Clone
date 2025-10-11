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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", '-d', type=str, default='cpu')
    parser.add_argument('--output-root', type=str, default=cfg.models_dir)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    outfile = os.path.join(args.output_root, f'{args.dataset}/{args.model}/{args.model}.pt')
    if args.skip_existing and os.path.exists(outfile):
        print(f'Output file {outfile} exists, skipping')
        sys.exit(0)

    ensure_dir_of(outfile)

    # Set num_workers=0 to avoid multiprocessing issues in Docker
    if args.dataset == 'IoTID20' and IOTID20_AVAILABLE:
        print("Using IoTID20 data preprocessing...")
        train_loader = get_benign_loader_iotid20(args.dataset, args.image_size, 'train', args.batch_size, shuffle=True, num_workers=0)
        val_loader = get_benign_loader_iotid20(args.dataset, args.image_size, 'test', args.batch_size, shuffle=True, num_workers=0)
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

    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):
        torch_model.train(True)
        for x, y in tqdm(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            y_pred = torch_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        torch_model.train(False)
        with torch.no_grad():
            val_acc = 0
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = torch_model(x)
                val_acc += (y_pred.argmax(dim=1) == y).sum().item()
            val_acc /= len(val_loader.dataset)
            print(f'Epoch {epoch}: {val_acc=}')

    torch.save(torch_model.state_dict(), outfile)
    print(f'Parameters saved to: {outfile}')

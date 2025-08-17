#!/usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}')

import torch
import torchvision
from tqdm import tqdm
import argparse
import cfg
from support import models
import dataman

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def train_model(model_name, dataset_name, epochs=10, batch_size=100, device='cpu'):
    """Train a model without TVM dependencies"""
    print(f"Training {model_name} on {dataset_name}...")
    
    outfile = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}.pt')
    ensure_dir_of(outfile)
    
    # Set num_workers=0 to avoid multiprocessing issues in Docker
    train_loader = dataman.get_benign_loader(dataset_name, 32, 'train', batch_size, shuffle=True, num_workers=0)
    val_loader = dataman.get_benign_loader(dataset_name, 32, 'test', batch_size, shuffle=True, num_workers=0)
    
    if dataset_name in {'ImageNet'}:
        model_class = getattr(torchvision.models, model_name)
    else:
        model_class = getattr(models, model_name)
    
    torch_model = model_class(pretrained=False)
    torch_model.to(device)
    
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(epochs)):
        torch_model.train(True)
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = torch_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        
        torch_model.train(False)
        with torch.no_grad():
            val_acc = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = torch_model(x)
                val_acc += (y_pred.argmax(dim=1) == y).sum().item()
            val_acc /= len(val_loader.dataset)
            print(f'Epoch {epoch}: {val_acc=}')
    
    torch.save(torch_model.state_dict(), outfile)
    print(f'Parameters saved to: {outfile}')
    return outfile

def test_model(model_name, dataset_name, batch_size=100, device='cpu'):
    """Test a trained model"""
    print(f"Testing {model_name} on {dataset_name}...")
    
    model_file = os.path.join(cfg.models_dir, f'{dataset_name}/{model_name}/{model_name}.pt')
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None
    
    # Set num_workers=0 to avoid multiprocessing issues in Docker
    test_loader = dataman.get_benign_loader(dataset_name, 32, 'test', batch_size, shuffle=False, num_workers=0)
    
    if dataset_name in {'ImageNet'}:
        model_class = getattr(torchvision.models, model_name)
    else:
        model_class = getattr(models, model_name)
    
    torch_model = model_class(pretrained=False)
    torch_model.load_state_dict(torch.load(model_file, map_location=device))
    torch_model.to(device)
    torch_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = torch_model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'test'], help='Action to perform')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()
    
    if args.action == 'train':
        train_model(args.model, args.dataset, args.epochs, args.batch_size, args.device)
    elif args.action == 'test':
        test_model(args.model, args.dataset, args.batch_size, args.device)

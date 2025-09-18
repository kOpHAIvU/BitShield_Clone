import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data_iotid20 import build_iotid20_loaders
from model_iotid20 import MLPClassifier
from custom_models import CustomModel1, CustomModel2

def build_model(model_name, in_features, n_classes):
    if model_name == 'mlp':
        return MLPClassifier(in_features, n_classes)
    elif model_name == 'custom1':
        return CustomModel1(input_size=in_features, output_size=n_classes)
    elif model_name == 'custom2':
        return CustomModel2(input_size=in_features, output_size=n_classes)
    else:
        raise ValueError('Unknown model: ' + model_name)


def train(train_loader, test_loader, in_features, n_classes, epochs=15, lr=1e-3, device='cpu', model_name='mlp'):
    model = build_model(model_name, in_features, n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = correct / total * 100
        test_acc = evaluate(model, test_loader, device)
        # Nicely formatted line
        print(f'[Epoch {epoch+1:03d}] Train: {train_acc:6.2f}% | Test: {test_acc:6.2f}% | Loss: {loss_sum/total:.4f}')
    return model


@torch.no_grad()
def evaluate(model, loader, device='cpu'):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--out', type=str, default='save/best_model.pth')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--source-csv', type=str, default=None)
    p.add_argument('--download-url', type=str, default=None)
    p.add_argument('--model', type=str, default='mlp', choices=['mlp', 'custom1', 'custom2'])
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train_loader, test_loader, in_features, n_classes = build_iotid20_loaders(args.data_root, source_csv=args.source_csv, download_url=args.download_url)
    model = train(train_loader, test_loader, in_features, n_classes, epochs=args.epochs, lr=args.lr, device=args.device, model_name=args.model)
    torch.save(model.state_dict(), args.out)
    print('Saved weights to', args.out)


if __name__ == '__main__':
    main()


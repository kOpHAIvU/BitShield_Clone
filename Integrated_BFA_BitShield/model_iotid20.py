import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, in_features, num_classes, hidden_sizes=(256, 128)):
        super().__init__()
        layers = []
        last = in_features
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.ReLU(inplace=True)]
            last = hs
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(last, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)


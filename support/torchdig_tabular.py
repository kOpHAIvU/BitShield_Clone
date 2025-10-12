# This file implements DIG for Tabular Data (IoTID20)
# Adapted from original DIG to work better with tabular/structured data

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np

class TabularDIGProtectedModule(nn.Module):
    """
    DIG Protection specifically designed for tabular data (IoTID20)
    
    Key improvements:
    1. Multiple detection metrics instead of just gradient norm
    2. Feature-wise analysis for tabular data
    3. Statistical anomaly detection
    4. Adaptive thresholds based on data distribution
    """
    
    def __init__(self, model, model_fc=None):
        super().__init__()
        self.model = model
        self.model_fc = model_fc or getattr(model, 'fc', getattr(model, 'classifier', None))
        assert self.model_fc is not None, 'No fc/classifier layer found'
        
        # Store statistics for adaptive thresholds
        self.feature_stats = None
        self.gradient_stats = None
        self.entropy_stats = None
        
    def forward(self, x):
        return self.model(x)
    
    def calc_sus_score(self, x):
        """
        Calculate comprehensive suspicious score for tabular data
        Combines multiple detection methods:
        1. Gradient norm (original DIG)
        2. Feature distribution anomaly
        3. Entropy-based detection
        4. Statistical outlier detection
        """
        logits = self.model(x)
        nclasses = logits.shape[1]
        self.model.zero_grad()
        
        # Method 1: Original gradient-based detection
        entropy = -1/nclasses * torch.sum(torch.log_softmax(logits, dim=1), dim=1)
        gradient = grad(entropy.sum(), self.model_fc.weight, create_graph=True)[0]
        gradient_norm = torch.abs(gradient).sum()
        
        # Method 2: Feature-wise anomaly detection
        feature_anomaly = self._calc_feature_anomaly(x)
        
        # Method 3: Entropy-based detection
        entropy_anomaly = self._calc_entropy_anomaly(logits)
        
        # Method 4: Statistical outlier detection
        statistical_anomaly = self._calc_statistical_anomaly(x, logits)
        
        # Combine all methods with weights
        combined_score = (
            0.4 * gradient_norm +           # Original DIG
            0.3 * feature_anomaly +         # Feature distribution
            0.2 * entropy_anomaly +         # Entropy analysis
            0.1 * statistical_anomaly       # Statistical outlier
        )
        
        return combined_score
    
    def _calc_feature_anomaly(self, x):
        """
        Detect anomalies in feature distribution
        For tabular data, we can analyze individual features
        """
        if self.feature_stats is None:
            return torch.tensor(0.0)
        
        # Calculate feature-wise z-scores
        feature_means = self.feature_stats['mean']
        feature_stds = self.feature_stats['std']
        
        # Avoid division by zero
        feature_stds = torch.where(feature_stds < 1e-8, torch.ones_like(feature_stds), feature_stds)
        
        z_scores = torch.abs((x - feature_means) / feature_stds)
        max_z_score = torch.max(z_scores)
        
        return max_z_score
    
    def _calc_entropy_anomaly(self, logits):
        """
        Detect anomalies in prediction entropy
        High entropy = uncertain predictions = potential attack
        """
        if self.entropy_stats is None:
            return torch.tensor(0.0)
        
        # Calculate entropy
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        avg_entropy = torch.mean(entropy)
        
        # Compare with normal entropy range
        normal_min, normal_max = self.entropy_stats['range']
        if avg_entropy < normal_min or avg_entropy > normal_max:
            return torch.tensor(1.0)  # High anomaly
        else:
            return torch.tensor(0.0)  # Normal
    
    def _calc_statistical_anomaly(self, x, logits):
        """
        Statistical outlier detection using multiple metrics
        """
        if self.gradient_stats is None:
            return torch.tensor(0.0)
        
        # Calculate prediction confidence
        probs = torch.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        avg_confidence = torch.mean(max_probs)
        
        # Compare with normal confidence range
        normal_min, normal_max = self.gradient_stats['confidence_range']
        if avg_confidence < normal_min or avg_confidence > normal_max:
            return torch.tensor(1.0)  # High anomaly
        else:
            return torch.tensor(0.0)  # Normal
    
    def update_statistics(self, clean_data_loader, device='cpu'):
        """
        Update statistics from clean data for adaptive thresholds
        This is crucial for tabular data as distributions can vary
        """
        print("Updating DIG statistics from clean data...")
        
        features = []
        entropies = []
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for x, _ in clean_data_loader:
                x = x.to(device)
                logits = self.model(x)
                
                # Collect features
                features.append(x.cpu())
                
                # Collect entropy statistics
                probs = torch.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropies.append(entropy.cpu())
                
                # Collect confidence statistics
                max_probs = torch.max(probs, dim=1)[0]
                confidences.append(max_probs.cpu())
        
        # Calculate feature statistics
        all_features = torch.cat(features, dim=0)
        self.feature_stats = {
            'mean': torch.mean(all_features, dim=0),
            'std': torch.std(all_features, dim=0)
        }
        
        # Calculate entropy statistics
        all_entropies = torch.cat(entropies, dim=0)
        entropy_mean = torch.mean(all_entropies)
        entropy_std = torch.std(all_entropies)
        self.entropy_stats = {
            'range': (entropy_mean - 2*entropy_std, entropy_mean + 2*entropy_std)
        }
        
        # Calculate confidence statistics
        all_confidences = torch.cat(confidences, dim=0)
        conf_mean = torch.mean(all_confidences)
        conf_std = torch.std(all_confidences)
        self.gradient_stats = {
            'confidence_range': (conf_mean - 2*conf_std, conf_mean + 2*conf_std)
        }
        
        print(f"Feature stats: mean={self.feature_stats['mean'][:5]}, std={self.feature_stats['std'][:5]}")
        print(f"Entropy range: {self.entropy_stats['range']}")
        print(f"Confidence range: {self.gradient_stats['confidence_range']}")

def wrap_with_tabular_dig(model, model_fc=None):
    """
    Wrap model with tabular DIG protection
    """
    return TabularDIGProtectedModule(model, model_fc)

def calc_tabular_dig_range(protected_model, train_loader, device='cpu', n_batches=20):
    """
    Calculate suspicious score range for tabular DIG
    """
    print("Calculating tabular DIG suspicious score range...")
    
    # Update statistics first
    protected_model.update_statistics(train_loader, device)
    
    # Calculate suspicious scores on clean data
    sus_scores = []
    protected_model.eval()
    
    batch_count = 0
    for x, _ in train_loader:
        if batch_count >= n_batches:
            break
            
        x = x.to(device)
        x.requires_grad_(True)
        
        try:
            sus_score = protected_model.calc_sus_score(x).item()
            sus_scores.append(sus_score)
        except Exception as e:
            print(f"Warning: Could not calculate suspicious score: {e}")
            sus_scores.append(0.0)
        
        x.requires_grad_(False)
        batch_count += 1
    
    sus_scores = np.array(sus_scores)
    
    # Use more conservative thresholds for tabular data
    min_score = np.percentile(sus_scores, 10)  # 10th percentile instead of 5th
    max_score = np.percentile(sus_scores, 90)  # 90th percentile instead of 95th
    
    print(f"Tabular DIG suspicious score range: [{min_score:.2f}, {max_score:.2f}]")
    print(f"Score statistics: mean={np.mean(sus_scores):.2f}, std={np.std(sus_scores):.2f}")
    
    return [min_score, max_score]

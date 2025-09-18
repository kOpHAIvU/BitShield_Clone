import os
import sys
import torch


def import_bitshield(bitshield_root):
    # Ensure both repo root and support/ are on sys.path so torchdig can import its utils
    repo_root = bitshield_root
    support_dir = os.path.join(bitshield_root, 'support')
    for p in [repo_root, support_dir]:
        if p not in sys.path:
            sys.path.append(p)
    import torchdig  # from BitShield_Clone/support/torchdig.py
    return torchdig


def wrap_with_dig(model, bitshield_root, model_fc=None):
    torchdig = import_bitshield(bitshield_root)
    protected = torchdig.DIGProtectedModule(model, model_fc=model_fc)
    return protected


def calc_dig_range(protected_model, loader, device='cpu', n_batches=50):
    import numpy as np
    sus_scores = []
    protected_model.to(device).eval()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        x.requires_grad_(True)
        try:
            s = protected_model.calc_sus_score(x).item()
            sus_scores.append(s)
        except Exception:
            pass
        x.requires_grad_(False)
    if not sus_scores:
        return [0.0, 0.0]
    sus_scores = np.array(sus_scores)
    return [float(np.percentile(sus_scores, 5)), float(np.percentile(sus_scores, 95))]


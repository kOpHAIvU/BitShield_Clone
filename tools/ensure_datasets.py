#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

from torchvision import datasets
from tqdm import tqdm
import lpips
import urllib.request

import cfg

def ensure_datasets(root=cfg.datasets_root):
    dataset_names = [
        'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'DTD', 'GTSRB'
    ]
    for dataset_name in tqdm(dataset_names):
        train_kwargs = {
            'DTD': {'split': 'train'},
            'GTSRB': {'split': 'train'},
        }.get(dataset_name, {'train': True})

        test_kwargs = {
            'DTD': {'split': 'test'},
            'GTSRB': {'split': 'test'},
        }.get(dataset_name, {'train': False})

        datasets.__dict__[dataset_name](
            root=root, download=True, **train_kwargs
        )
        datasets.__dict__[dataset_name](
            root=root, download=True, **test_kwargs
        )
    lpips.LPIPS()

    datasets.imagenet.ARCHIVE_META.update({  # type: ignore
        "train": ("ILSVRC2012_img_train.tar", None),
        "val": ("ILSVRC2012_img_val.tar", None),
    })
    [datasets.ImageNet(cfg.imagenet_root, split=s) for s in ['train', 'val']]

    # Torchvision models pretrained on ImageNet
    # all_models = {bi.model_name for bi in cfg.all_build_bis}
    # [getattr(tv_models, m)(pretrained=True) for m in tqdm(all_models)]

    fid_weights_path = f'{cfg.models_dir}/ImageNet/FIDInception/FIDInception.pt'
    if not os.path.exists(fid_weights_path):
        os.makedirs(os.path.dirname(fid_weights_path), exist_ok=True)
        urllib.request.urlretrieve('https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth', fid_weights_path)

if __name__ == '__main__':
    ensure_datasets()

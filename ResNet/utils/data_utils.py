from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


def get_transforms(augment: bool = True, normalize: bool = True):
    t_list = []
    if augment:
        t_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    t_list.append(transforms.ToTensor())
    if normalize:
        t_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    train_transform = transforms.Compose(t_list)

    test_t_list = [transforms.ToTensor()]
    if normalize:
        test_t_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    test_transform = transforms.Compose(test_t_list)

    return train_transform, test_transform

def subsample_dataset(
    dataset: Dataset,
    train_size: Optional[int],
    seed: int,
) -> Dataset:
    if train_size is None or train_size >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:train_size]
    return Subset(dataset, indices)


def get_mnist_transforms(normalize: bool = True):
    train_t = [transforms.ToTensor()]
    test_t  = [transforms.ToTensor()]

    if normalize:
        train_t.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
        test_t.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))

    train_transform = transforms.Compose(train_t)
    test_transform  = transforms.Compose(test_t)
    return train_transform, test_transform



def apply_symmetric_label_noise(dataset, noise_fraction: float, num_classes: int, seed: int = 0):

    if noise_fraction <= 0.0:
        return

    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    n = len(targets)
    num_noisy = int(noise_fraction * n)

    noisy_indices = rng.choice(n, size=num_noisy, replace=False)
    for idx in noisy_indices:
        old = targets[idx]
        new = rng.randint(num_classes - 1)
        if new >= old:
            new += 1
        targets[idx] = new

    dataset.targets = targets.tolist()


def get_cifar10_dataloaders(
    data_root: str,
    batch_size: int = 128,
    noise_fraction: float = 0.15,
    augment: bool = True,
    normalize: bool = True,
    num_workers: int = 4,
    seed: int = 0,
    train_size: int = 25000
):
    train_tf, test_tf = get_transforms(augment=augment, normalize=normalize)

    train_set = datasets.CIFAR10(root=data_root, train=True, download=False,
                                 transform=train_tf)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=False,
                                transform=test_tf)

  
    apply_symmetric_label_noise(train_set, noise_fraction, num_classes=10, seed=seed)

    train_dataset = subsample_dataset(train_set, train_size, seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def get_mnist_dataloaders(
    data_root: str,
    batch_size: int = 128,
    noise_fraction: float = 0.0,   
    normalize: bool = True,
    num_workers: int = 4,
    seed: int = 0,
    train_size: int = 30000
):

    train_tf, test_tf = get_mnist_transforms(normalize=normalize)

    train_set = datasets.MNIST(root=data_root, train=True, download=False,
                               transform=train_tf)
    test_set  = datasets.MNIST(root=data_root, train=False, download=False,
                               transform=test_tf)
    
    train_dataset = subsample_dataset(train_set, train_size, seed)

    apply_symmetric_label_noise(train_set, noise_fraction, num_classes=10, seed=seed)

    train_dataset = subsample_dataset(train_set, train_size, seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader

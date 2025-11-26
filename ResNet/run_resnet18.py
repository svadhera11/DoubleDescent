import argparse
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loguru import logger
from tqdm import tqdm


from models.resnet import PreActResNet18k
from utils.data_utils import get_cifar10_dataloaders, get_mnist_dataloaders
from utils.experiment_utils import set_seed, train_one_epoch, evaluate
from utils.model_utils import build_criterion, build_optimizer

def run_experiment(
    data_root: str = "./data",
    k: int = 64,
    noise_fraction: float = 0.15,
    epochs: int = 4000,
    lr: float = 1e-4,
    batch_size: int = 128,
    device: str = None,
    seed: int = 0,
    tolerance: Optional[int] = None,
    num_workers: int = 4,
    optimizer_name: str = "Adam",
    loss_name: str = "ce",
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    dataset: str = "cifar10"
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            noise_fraction=noise_fraction,
            augment=True,
            normalize=True,
            seed=seed,
            num_workers=num_workers
        )
        model = PreActResNet18k(
            k = k, 
            num_classes = 10,
            in_channels = 3
        ).to(device)
    elif dataset == "mnist":
        train_loader, test_loader = get_mnist_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            noise_fraction=noise_fraction,
            normalize=True,
            seed=seed,
            num_workers=num_workers
        )
        model = PreActResNet18k(
            k = k, 
            num_classes = 10,
            in_channels = 1
        ).to(device)
    else:
        raise NotImplementedError

    criterion = build_criterion(loss_name = loss_name)
    optimizer = build_optimizer(
        opt_name = optimizer_name,
        model_params = model.parameters(),
        lr = lr,
        weight_decay = weight_decay,
        momentum = momentum
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: {}(k={}, num_classes={}) | params: total={:,}, trainable={:,}",
        model.__class__.__name__, k, 10, n_params, n_trainable,
    )
    logger.info("Optimizer: {}", optimizer.__class__.__name__)
    for idx, g in enumerate(optimizer.param_groups):
        logger.info(
            "  param_group {} | lr={} | weight_decay={} | betas={} | eps={}",
            idx,
            g.get("lr", None),
            g.get("weight_decay", 0.0),
            g.get("betas", None),
            g.get("eps", None),
        )

    logger.info("Loss: {}", criterion.__class__.__name__)
    logger.info("Running on device: {}", device)

    best_test_loss = float("inf")
    best_test_acc = 0.0
    best_epoch = 0
    epochs_since_best = 0
    best_state_dict = None

    eval_every = 50

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % eval_every == 0 or epoch == 1: 
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            improved = test_loss < best_test_loss - 1e-12  # strict improvement
            if improved:
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_epoch = epoch
                epochs_since_best = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_since_best += 1

        if epoch % 200 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc*100:.2f}%, "
                f"best_test_loss={best_test_loss:.4f} (epoch {best_epoch})"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    logger.info(
        f"Best test performance: loss={best_test_loss:.4f}, "
        f"acc={best_test_acc*100:.2f}% at epoch {best_epoch}"
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--log-dir", type = str, default = "logs")
    parser.add_argument("--k", type=int, default=64,
                        help="width parameter; standard ResNet18 ≈ 64")
    parser.add_argument("--noise", type=float, default=0.15,
                        help="fraction of label noise (0.0–0.2 in the paper)")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type = int, default = 4)
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "sgd_mom"])
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dataset", type=str, default="cifar10")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_dir = f"{args.log_dir}/resnet18/k{args.k}_noise{int(100 * args.noise)}_{args.dataset}_optim{args.optimizer}_loss{args.loss}.log"
    logger.add(log_dir, level="INFO", enqueue=True)
    logger.info(args)

    run_experiment(
        data_root=args.data_root,
        k=args.k,
        noise_fraction=args.noise,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        tolerance=args.tolerance,
        num_workers=args.num_workers,
        dataset = args.dataset
    )

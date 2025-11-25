import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4) 
torch.set_num_interop_threads(4)
torch.backends.cudnn.benchmark = True

class CNN5(nn.Module):
    def __init__(self, width: int, in_channels: int, num_classes: int):
        super().__init__()
        k1, k2, k3, k4 = width, 2 * width, 4 * width, 8 * width
        self.conv1 = nn.Conv2d(in_channels, k1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(k1)
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(k2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(k2, k3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(k3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(k3, k4, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(k4)
        self.pool4 = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(k4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


class NoisyLabels(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        num_classes: int,
        noise_prob: float,
        seed: int = 0,
    ):
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.noise_prob = noise_prob

        rng = random.Random(seed)
        if hasattr(base_dataset, "targets"):
            base_targets = list(base_dataset.targets)
        elif hasattr(base_dataset, "labels"):
            base_targets = list(base_dataset.labels)
        else:
            raise ValueError("Base dataset has no 'targets' or 'labels' attribute.")

        noisy_targets = []
        for y in base_targets:
            if rng.random() < noise_prob:
                new_label = rng.randrange(num_classes - 1)
                if new_label >= y:
                    new_label += 1
                noisy_targets.append(new_label)
            else:
                noisy_targets.append(y)

        self.targets = noisy_targets

    def __getitem__(self, idx: int):
        x, _ = self.base_dataset[idx]
        y = self.targets[idx]
        if not isinstance(x, torch.Tensor):
            x = transforms.functional.to_tensor(x)
        y = torch.as_tensor(y, dtype=torch.long)
        return x, y

    def __len__(self) -> int:
        return len(self.base_dataset)


def get_transforms(
    dataset_name: str,
    train: bool,
    augment: bool,
) -> transforms.Compose:
    name = dataset_name.lower()

    if name in ("cifar10", "cifar100"):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        t_list = []
        if train and augment:
            t_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Normalize(mean, std))
        return transforms.Compose(t_list)

    elif name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
        t_list = []
        if train and augment:
            t_list.extend([
                transforms.RandomCrop(28, padding=2),
            ])
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Normalize(mean, std))
        return transforms.Compose(t_list)

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


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


def get_datasets(
    dataset_name: str,
    data_root: str,
    noise_prob: float,
    augment: bool,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, int, int]:
    name = dataset_name.lower()
    if name == "cifar10":
        train_base = datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=get_transforms(name, train=True, augment=augment),
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=get_transforms(name, train=False, augment=False),
        )
        in_channels = 3
        num_classes = 10

    elif name == "cifar100":
        train_base = datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=get_transforms(name, train=True, augment=augment),
        )
        test_dataset = datasets.CIFAR100(
            root=data_root, train=False, download=True,
            transform=get_transforms(name, train=False, augment=False),
        )
        in_channels = 3
        num_classes = 100

    elif name == "mnist":
        train_base = datasets.MNIST(
            root=data_root, train=True, download=True,
            transform=get_transforms(name, train=True, augment=augment),
        )
        test_dataset = datasets.MNIST(
            root=data_root, train=False, download=True,
            transform=get_transforms(name, train=False, augment=False),
        )
        in_channels = 1
        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if noise_prob > 0.0:
        train_dataset = NoisyLabels(
            base_dataset=train_base,
            num_classes=num_classes,
            noise_prob=noise_prob,
            seed=seed,
        )
    else:
        train_dataset = train_base

    return train_dataset, test_dataset, in_channels, num_classes


def get_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    noise_prob: float,
    augment: bool,
    seed: int = 0,
    train_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    train_dataset, test_dataset, in_channels, num_classes = get_datasets(
        dataset_name=dataset_name,
        data_root=data_root,
        noise_prob=noise_prob,
        augment=augment,
        seed=seed,
    )

    train_dataset = subsample_dataset(train_dataset, train_size, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader, in_channels, num_classes


@dataclass
class TrainConfig:
    dataset: str = "cifar10"
    data_root: str = "./data"
    batch_size: int = 128
    noise_prob: float = 0.2          
    augment: bool = True             
    width: int = 64                  
    num_steps: int = 500_000         
    eval_every: int = 10_000         
    lr0: float = 0.1                 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    train_size: Optional[int] = None


def inverse_sqrt_lr(step: int, lr0: float, decay_every: int = 512) -> float:
    return lr0 / math.sqrt(1.0 + (step // decay_every))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    error = 1.0 - accuracy
    return {"loss": avg_loss, "acc": accuracy, "err": error}


def train_model(config: TrainConfig) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)


    train_loader, test_loader, in_channels, num_classes = get_dataloaders(
        dataset_name=config.dataset,
        data_root=config.data_root,
        batch_size=config.batch_size,
        noise_prob=config.noise_prob,
        augment=config.augment,
        seed=config.seed,
        train_size=config.train_size,
    )

    model = CNN5(
        width=config.width,
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(config.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr0,
        momentum=0.0,
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "steps": [],
        "train_loss": [],
        "train_err": [],
        "test_loss": [],
        "test_err": [],
    }

    model.train()
    train_iter = iter(train_loader)

    step_iter = trange(
        config.num_steps,
        desc=f"Train {config.dataset}, k={config.width}, p={config.noise_prob}, n={config.train_size}",
        leave=False,
    )

    for s in step_iter:
        step = s + 1 

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(config.device, non_blocking=True)
        y = y.to(config.device, non_blocking=True)

        lr = inverse_sqrt_lr(step - 1, config.lr0)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == 1 or step == config.num_steps:
            model.eval()
            with torch.no_grad():
                logits_train = model(x)
                loss_train = criterion(logits_train, y).item()
                preds_train = logits_train.argmax(dim=1)
                train_err = 1.0 - (preds_train == y).float().mean().item()

            test_metrics = evaluate(model, test_loader, config.device)
            history["steps"].append(step)
            history["train_loss"].append(loss_train)
            history["train_err"].append(train_err)
            history["test_loss"].append(test_metrics["loss"])
            history["test_err"].append(test_metrics["err"])

            model.train()

    final_test = evaluate(model, test_loader, config.device)
    history["final_test"] = final_test
    history["final_train_err"] = history["train_err"][-1] if history["train_err"] else None
    history["config"] = config
    return history


def run_model_wise_grid(
    dataset: str,
    widths: List[int],
    noise_probs: List[float],
    seeds: List[int],
    train_size: Optional[int] = None,
    augment: bool = True,
    num_steps: int = 500_000,
    eval_every: int = 10_000,
    lr0: float = 0.1,
    batch_size: int = 128,
    data_root: str = "./data",
) -> Dict[str, Any]:

    results: Dict[float, Dict[int, Any]] = {}

    for p in tqdm(noise_probs, desc=f"Noise levels ({dataset}, n={train_size})"):
        results[p] = {}
        for w in tqdm(widths, desc=f"Widths (p={p})", leave=False):
            test_errs = []
            train_errs = []
            histories = []

            for seed in tqdm(seeds, desc=f"Seeds (k={w})", leave=False):
                cfg = TrainConfig(
                    dataset=dataset,
                    data_root=data_root,
                    batch_size=batch_size,
                    noise_prob=p,
                    augment=augment,
                    width=w,
                    num_steps=num_steps,
                    eval_every=eval_every,
                    lr0=lr0,
                    seed=seed,
                    train_size=train_size,
                )
                hist = train_model(cfg)
                histories.append(hist)
                test_errs.append(hist["final_test"]["err"])
                train_errs.append(hist["final_train_err"])

            test_errs = np.array(test_errs)
            train_errs = np.array(train_errs)

            results[p][w] = {
                "mean_test_err": float(test_errs.mean()),
                "std_test_err": float(test_errs.std(ddof=0)),
                "mean_train_err": float(train_errs.mean()),
                "std_train_err": float(train_errs.std(ddof=0)),
                "histories": histories,
            }

    return results


def run_epoch_wise_multi(
    dataset: str,
    width: int,
    noise_prob: float,
    seeds: List[int],
    train_size: Optional[int] = None,
    augment: bool = True,
    num_steps: int = 500_000,
    eval_every: int = 1_000,
    lr0: float = 0.1,
    batch_size: int = 128,
    data_root: str = "./data",
) -> Dict[str, Any]:

    histories = []
    steps_ref = None
    train_errs = []
    test_errs = []
    train_losses = []
    test_losses = []

    for seed in tqdm(seeds, desc=f"Epoch-wise seeds (k={width}, p={noise_prob}, n={train_size})"):
        cfg = TrainConfig(
            dataset=dataset,
            data_root=data_root,
            batch_size=batch_size,
            noise_prob=noise_prob,
            augment=augment,
            width=width,
            num_steps=num_steps,
            eval_every=eval_every,
            lr0=lr0,
            seed=seed,
            train_size=train_size,
        )
        hist = train_model(cfg)
        histories.append(hist)

        if steps_ref is None:
            steps_ref = hist["steps"]
        else:
            assert steps_ref == hist["steps"], "Mismatched eval steps across seeds"

        train_errs.append(hist["train_err"])
        test_errs.append(hist["test_err"])
        train_losses.append(hist["train_loss"])
        test_losses.append(hist["test_loss"])

    train_errs = np.array(train_errs)
    test_errs = np.array(test_errs)
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    result = {
        "steps": steps_ref,
        "train_err_mean": train_errs.mean(axis=0).tolist(),
        "train_err_std": train_errs.std(axis=0, ddof=0).tolist(),
        "test_err_mean": test_errs.mean(axis=0).tolist(),
        "test_err_std": test_errs.std(axis=0, ddof=0).tolist(),
        "train_loss_mean": train_losses.mean(axis=0).tolist(),
        "train_loss_std": train_losses.std(axis=0, ddof=0).tolist(),
        "test_loss_mean": test_losses.mean(axis=0).tolist(),
        "test_loss_std": test_losses.std(axis=0, ddof=0).tolist(),
        "histories": histories,
    }
    return result


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def plot_model_wise_results(
    results: Dict[float, Dict[int, Any]],
    widths: List[int],
    noise_probs: List[float],
    out_dir: str,
    tag: str = "",
) -> None:

    ensure_dir(out_dir)

    pickle_path = os.path.join(out_dir, f"modelwise_{tag or 'results'}.pkl")
    save_pickle(results, pickle_path)
    print(f"Saved model-wise results to {pickle_path}")

    plt.figure()
    for p in noise_probs:
        means = [results[p][w]["mean_test_err"] for w in widths]
        stds = [results[p][w]["std_test_err"] for w in widths]
        plt.errorbar(
            widths,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            label=f"p={p:.1f}",
        )
    plt.xscale("log", base=2)
    plt.xlabel("Width k")
    plt.ylabel("Test error")
    plt.title(f"Model-wise double descent (test error){' - ' + tag if tag else ''}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    fig_path = os.path.join(out_dir, f"modelwise_test_error_{tag or 'results'}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved model-wise test-error plot to {fig_path}")

    plt.figure()
    for p in noise_probs:
        means = [results[p][w]["mean_train_err"] for w in widths]
        stds = [results[p][w]["std_train_err"] for w in widths]
        plt.errorbar(
            widths,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            label=f"p={p:.1f}",
        )
    plt.xscale("log", base=2)
    plt.xlabel("Width k")
    plt.ylabel("Train error")
    plt.title(f"Model-wise train error{' - ' + tag if tag else ''}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    fig_path = os.path.join(out_dir, f"modelwise_train_error_{tag or 'results'}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved model-wise train-error plot to {fig_path}")


def plot_epoch_wise_result(
    epoch_result: Dict[str, Any],
    out_dir: str,
    tag: str = "",
) -> None:
    ensure_dir(out_dir)


    pickle_path = os.path.join(out_dir, f"epochwise_{tag or 'results'}.pkl")
    save_pickle(epoch_result, pickle_path)
    print(f"Saved epoch-wise results to {pickle_path}")

    steps = epoch_result["steps"]
    steps = np.array(steps, dtype=float)


    test_mean = np.array(epoch_result["test_err_mean"])
    test_std = np.array(epoch_result["test_err_std"])
    train_mean = np.array(epoch_result["train_err_mean"])
    train_std = np.array(epoch_result["train_err_std"])

    plt.figure()
    plt.plot(steps, test_mean, label="test error (mean)")
    plt.fill_between(
        steps,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.3,
        label="test error ±1 std",
    )
    plt.plot(steps, train_mean, label="train error (mean)")
    plt.fill_between(
        steps,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.3,
        label="train error ±1 std",
    )
    plt.xlabel("Training step")
    plt.ylabel("Error")
    plt.title(f"Epoch-wise double descent (error){' - ' + tag if tag else ''}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    fig_path = os.path.join(out_dir, f"epochwise_error_{tag or 'results'}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved epoch-wise error plot to {fig_path}")


    test_loss_mean = np.array(epoch_result["test_loss_mean"])
    test_loss_std = np.array(epoch_result["test_loss_std"])
    train_loss_mean = np.array(epoch_result["train_loss_mean"])
    train_loss_std = np.array(epoch_result["train_loss_std"])

    plt.figure()
    plt.plot(steps, test_loss_mean, label="test loss (mean)")
    plt.fill_between(
        steps,
        test_loss_mean - test_loss_std,
        test_loss_mean + test_loss_std,
        alpha=0.3,
        label="test loss ±1 std",
    )
    plt.plot(steps, train_loss_mean, label="train loss (mean)")
    plt.fill_between(
        steps,
        train_loss_mean - train_loss_std,
        train_loss_mean + train_loss_std,
        alpha=0.3,
        label="train loss ±1 std",
    )
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Epoch-wise loss{' - ' + tag if tag else ''}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    fig_path = os.path.join(out_dir, f"epochwise_loss_{tag or 'results'}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved epoch-wise loss plot to {fig_path}")

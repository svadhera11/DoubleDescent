import torch.nn as nn
import torch.optim as optim

def build_criterion(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name in ["ce", "cross_entropy", "cross-entropy"]:
        return nn.CrossEntropyLoss()
    elif loss_name in ["mse", "l2"]:
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")


def build_optimizer(
    opt_name: str,
    model_params,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
):
    opt_name = opt_name.lower()

    if opt_name == "adam":
        # matches paper: Adam, default betas, eps, weight_decay=0
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)

    elif opt_name == "sgd":
        # plain SGD, no momentum
        return optim.SGD(model_params, lr=lr, momentum=0.0, weight_decay=weight_decay)

    elif opt_name in ["sgd_mom", "sgd-mom", "sgdm"]:
        # SGD with momentum, commonly used alternative
        return optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
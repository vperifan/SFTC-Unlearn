import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from utils.train_utils import base_fit


def catastrophic_forgetting_k(
        model: torch.nn.Module,
        retain_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        forget_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: torch.nn.Module,
        epochs: int = 3,
        return_history: bool = False,
        device: str = 'cuda',
        verbose: bool = True,
        **kwargs
):
    """
    CF-k Unlearning: Freeze the first k layers of the original model and finetune the remaining layers.
    EU-k Unlearning: Freeze the first k layers of the original model, randomly initialize the rest layers and finetune.

    Goel, S., Prabhu, A., Sanyal, A., Lim, S. N., Torr, P., & Kumaraguru, P. (2022).
    Towards adversarial evaluations for inexact machine unlearning. arXiv preprint arXiv:2201.06640.
    """
    eu_k = kwargs.get("eu_k", False)
    k = kwargs.get("k", 7)

    model.train()
    layers = list(model.children())

    if model.__class__.__name__.lower() == "resnet":
        for i in range(k):
            for param in layers[i].parameters():
                param.requires_grad = False
        if verbose:
            for name, param in model.named_parameters():
                print(name, param.requires_grad)
        if eu_k:
            for i in range(k, len(layers)):
                for m in layers[i].modules():
                    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                        torch.nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)
    elif model.__class__.__name__.lower() == "efficientnet":
        features = model.features
        for i in range(k):
            for param in features[i].parameters():
                param.requires_grad = False
        if verbose:
            for name, param in model.named_parameters():
                print(name, param.requires_grad)
        if eu_k:
            for i in range(k, len(features)):
                for m in features[i].modules():
                    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                        torch.nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)
    else:
        raise ValueError

    if return_history:
        model, history = base_fit(
            model, retain_loader, val_loader, test_loader, forget_loader, optimizer, scheduler,
            criterion, epochs, return_history, device
        )
        return model, history

    else:
        model = base_fit(
            model, retain_loader, val_loader, test_loader, forget_loader, optimizer, scheduler,
            criterion, epochs, return_history, device
        )
        return model

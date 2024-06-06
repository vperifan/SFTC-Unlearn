import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from utils.train_utils import base_fit


def fine_tune(
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
):
    """
    Simple forgetting by fine-tuning on the retain set.
    """
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

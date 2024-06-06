import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import predict_epoch


def neg_grad(model: torch.nn.Module,
             retain_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: DataLoader,
             forget_loader: DataLoader,
             optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.LRScheduler,
             criterion: torch.nn.Module,
             epochs: int = 5,
             return_history: bool = False,
             device: str = 'cuda',
             **kwargs
             ):
    advanced_neg_grad = kwargs.get("advanced_neg_grad", False)

    train_losses, val_losses, test_losses, forget_losses = [], [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    retain_iterator = iter(retain_loader)
    for epoch in range(epochs):
        model.train()

        running_loss = []
        for x_forget, y_forget in tqdm(forget_loader, desc=f"Epoch {epoch + 1} - Training"):
            x_forget, y_forget = x_forget.to(device), y_forget.to(device)

            try:
                x, y = next(retain_iterator)
            except StopIteration:
                retain_iterator = iter(retain_loader)
                x, y = next(retain_iterator)

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # make a prediction on the forget set
            outputs_forget = model(x_forget)
            total_loss = -criterion(outputs_forget, y_forget)

            if advanced_neg_grad:
                # make a prediction on the retain set
                outputs = model(x)
                total_loss += criterion(outputs, y)

            total_loss.backward()
            optimizer.step()
            running_loss.append(total_loss.item())
        if scheduler is not None:
            scheduler.step()
        epoch_loss = sum(running_loss) / len(running_loss)
        train_losses.append(epoch_loss)
        _, train_acc = predict_epoch(model, retain_loader, criterion, device)
        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\tLoss={epoch_loss:.4f}, Acc={train_acc:.4f}\n\t"
              f"[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")

    if return_history:
        losses = {"train": train_losses,
                  "val": val_losses,
                  "test": test_losses}
        accs = {"train": train_accs,
                "val": val_accs,
                "test": test_accs}
        if forget_loader is not None:
            losses["forget"] = forget_losses
            accs["forget"] = forget_accs

        return model, [losses, accs]
    return model

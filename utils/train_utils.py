from typing import Union

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm


def base_fit(model: torch.nn.Module,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             test_loader: torch.utils.data.DataLoader,
             forget_loader: Union[None, torch.utils.data.DataLoader],
             optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.LRScheduler,
             criterion: torch.nn.Module,
             epochs: int = 30,
             return_history: bool = False,
             device: str = 'cuda'
             ):
    train_losses, val_losses, test_losses, forget_losses = [], [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()

        running_loss = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        epoch_loss = sum(running_loss) / len(running_loss)
        train_losses.append(epoch_loss)
        train_acc = 0.
        _, train_acc = predict_epoch(model, train_loader, criterion, device)
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


def predict_epoch(
        model,
        data_loader,
        criterion,
        device='cuda'
):
    model.eval()
    running_loss = []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    loss = sum(running_loss) / len(running_loss)
    accuracy = correct / total

    return loss, accuracy

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import predict_epoch
from utils.loss_utils import custom_kl_loss


def bad_teaching(
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
        **kwargs
):
    try:
        teacher_model = kwargs['teacher_model']
    except KeyError:
        raise ValueError("Bad Teaching requires the teacher_model to be given as input!")
    try:
        dummy_model = kwargs['dummy_model']
    except KeyError:
        raise ValueError("Bad Teaching requires the dummy_model to be given as input!")

    temperature = kwargs.get("temperature", 4.)

    teacher_model.eval()
    dummy_model.eval()

    model.train()

    train_losses_retain, train_losses_forget = [], []
    val_losses, test_losses, forget_losses = [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    # Training with retain data
    for epoch in range(epochs):
        model.train()

        running_kl_retain = []
        for x, y, pseudo_label in tqdm(retain_loader, desc=f"Epoch {epoch + 1} - Training"):
            x, y, pseudo_label = x.to(device), y.to(device), pseudo_label.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(x)
                dummy_outputs = dummy_model(x)

            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x)

            loss = custom_kl_loss(teacher_logits=teacher_outputs,
                                  dummy_logits=dummy_outputs,
                                  student_logits=outputs,
                                  pseudo_labels=pseudo_label,
                                  kl_temperature=temperature)
            running_kl_retain.append(loss.item())
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_kl_retain_loss = sum(running_kl_retain) / len(running_kl_retain)

        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\t"
              f"Retain KL Loss={epoch_kl_retain_loss:.4f}\n\t"
              f"[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")

    if return_history:
        losses = {"train": train_losses_retain,
                  "val": val_losses,
                  "test": test_losses}
        accs = {
            "val": val_accs,
            "test": test_accs}
        if forget_loader is not None:
            losses["forget"] = forget_losses
            accs["forget"] = forget_accs

        return model, [losses, accs]
    return model

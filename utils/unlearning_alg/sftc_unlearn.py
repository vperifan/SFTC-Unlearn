import copy
import sys
from pathlib import Path

from utils.unlearning_utils import RandomDistributionGenerator

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import predict_epoch
from utils.loss_utils import custom_kl_loss, SelectiveCrossEntropyLoss, kl_loss


def sftc_unlearn(
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
        merged_loader = kwargs['merged_loader']
    except KeyError:
        raise ValueError("SFTC requires the merged loader to be given as input!")
    try:
        teacher_model = kwargs['teacher_model']
    except KeyError:
        raise ValueError("SFTC requires the teacher_model to be given as input!")
    try:
        dummy_model = kwargs['dummy_model']
    except KeyError:
        raise ValueError("SFTC requires the dummy_model to be given as input!")

    temperature = kwargs.get("temperature", 4.)
    confuse_fraction = kwargs.get("confuse_fraction", 0.)

    labels = set()
    for _, label in val_loader:
        labels.update(label.numpy())
    num_classes = len(labels)

    selective_criterion = SelectiveCrossEntropyLoss()

    teacher_model.eval()
    if not isinstance(dummy_model, RandomDistributionGenerator):
        dummy_model.eval()

    model.train()

    train_losses_retain, train_losses_forget = [], []
    val_losses, test_losses, forget_losses = [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    # Training with retain data
    for epoch in range(epochs):
        model.train()
        running_cross_entropy_retain = []
        running_kl = []
        for x, y, pseudo_label in tqdm(merged_loader, desc=f"Epoch {epoch + 1} - Training"):
            x, y, pseudo_label = x.to(device), y.to(device), pseudo_label.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(x)

                # Confusion
                if confuse_fraction == 0.:
                    if isinstance(dummy_model, RandomDistributionGenerator):
                        dummy_outputs = dummy_model(len(x), y)
                    else:
                        raise ValueError("Cannot perform confusion with fraction=0. and a trained model!"
                                         " Please use confuse_fraction=1.")
                elif confuse_fraction == 1.:
                    if isinstance(dummy_model, RandomDistributionGenerator):
                        dummy_outputs = dummy_model(len(x), None)
                        dummy_outputs = dummy_outputs.to(device)
                    else:
                        dummy_outputs = dummy_model(x)
                else:
                    # Select indices where pseudo_label equals to 1
                    forget_indices = torch.where(pseudo_label == 1)[0]
                    # Make a random permutation
                    permuted_indices = forget_indices[torch.randperm(len(forget_indices))]
                    n = len(forget_indices)
                    replace_n = int(n * confuse_fraction)
                    if replace_n == 0:
                        replace_n = 1
                    fake_indices = torch.randperm(n)[:replace_n]
                    indices = permuted_indices[fake_indices]
                    tmp_y = copy.deepcopy(y)
                    tmp_y[indices] = torch.randint(0, num_classes, (replace_n,), device=y.device)

                    if isinstance(dummy_model, RandomDistributionGenerator):
                        dummy_outputs = dummy_model(len(x), tmp_y)
                    else:
                        raise ValueError(f"Cannot perform confusion with fraction={confuse_fraction} "
                                         f"and a trained model! Please use confuse_fraction=1.")

            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x)

            selective_cross_entropy = selective_criterion(outputs, y, pseudo_label, 0)

            kl_div = custom_kl_loss(
                teacher_logits=teacher_outputs,
                dummy_logits=dummy_outputs,
                student_logits=outputs,
                pseudo_labels=pseudo_label,
                kl_temperature=temperature
            )
            running_cross_entropy_retain.append(selective_cross_entropy.item())
            running_kl.append(kl_div.item())

            total_loss = selective_cross_entropy + kl_div
            total_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_cross_entropy = sum(running_cross_entropy_retain) / len(running_cross_entropy_retain)
        epoch_kl_retain_loss = sum(running_kl) / len(running_kl)

        _, train_acc = predict_epoch(model, retain_loader, criterion, device)
        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        train_losses_retain.append(epoch_cross_entropy)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\t"
              f"Retain Loss={epoch_cross_entropy:.4f}, Retain Acc={train_acc:.4f}, KL Loss={epoch_kl_retain_loss:.4f}\n"
              f"\t[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")

    if return_history:
        losses = {"train": train_losses_retain,
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
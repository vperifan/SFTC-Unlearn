import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Dict

import random
import pandas as pd
from scipy.stats import skew, kurtosis

import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def predict(model, data_loader, device):
    model.eval()
    running_loss = []
    correct, total = 0, 0
    criterion = torch.nn.CrossEntropyLoss()

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
    acc = correct / total

    return {"Loss": loss, "Acc": acc}


def predict_all(models: List[torch.nn.Module], model_names: List[str], dataloaders: Dict[str, DataLoader], device: str):
    outputs = dict()
    for model, model_name in zip(models, model_names):
        outputs[model_name] = dict()
        for data_name, data_loader in dataloaders.items():
            output = predict(model, data_loader, device)
            outputs[model_name][data_name] = output
    return outputs


def pretty_predict_all(models: List[torch.nn.Module],
                       model_names: List[str],
                       dataloaders: Dict[str, DataLoader],
                       device: str):
    outputs = predict_all(models, model_names, dataloaders, device)
    rows = []
    for model_type, model_data in outputs.items():
        for dataset_type, dataset_data in model_data.items():
            row = {'Model': model_type, 'Dataset': dataset_type}
            row.update(dataset_data)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(by='Dataset')
    return df


@torch.no_grad()
def get_confidences(model, loader, device):
    model.eval()
    confidences = []

    for inputs, _ in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        logits_prob = F.softmax(logits, dim=1)
        confidences.extend(logits_prob.tolist())
    return confidences


def conf_stats_all(models, model_names, loaders, device):
    outputs = dict()
    for model, model_name in zip(models, model_names):
        outputs[model_name] = dict()
        for loader_name, loader in loaders.items():
            confs = get_confidences(model, loader, device)
            confs_tensor: torch.Tensor = torch.Tensor(confs)
            max_probs = np.max(confs, 1)
            outputs[model_name][loader_name] = {
                "Mean": np.mean(max_probs),
                "Std": np.std(max_probs),
                "Skew": skew(max_probs, bias=False),
                "Kurtosis": kurtosis(max_probs, bias=False),
                "Mean Entropy": float(torch.mean(torch.sum(-confs_tensor * torch.log(confs_tensor), dim=1)))}

    return outputs


def pretty_conf_stats_all(models, model_names, loaders, device):
    outputs = conf_stats_all(models, model_names, loaders, device)
    rows = []
    for model_type, model_data in outputs.items():
        for dataset_type, dataset_data in model_data.items():
            row = {'Model': model_type, 'Dataset': dataset_type}
            row.update(dataset_data)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(by='Dataset')
    return df


def sape(a, b):
    return abs(b - a) / (b + a) # * 100


def js_ind(unlearned_model, original_model, forget_loader, device):
    unlearned_model.eval()
    original_model.eval()
    unlearned_model_preds, original_model_preds = [], []
    with torch.no_grad():
        for x, _ in forget_loader:
            x = x.to(device)
            unlearned_model_out = unlearned_model(x)
            original_model_out = original_model(x)
            unlearned_model_preds.append(F.softmax(unlearned_model_out, dim=1).cpu())
            original_model_preds.append(F.softmax(original_model_out, dim=1).cpu())

    unlearned_model_preds = torch.cat(unlearned_model_preds, dim=0)
    original_model_preds = torch.cat(original_model_preds, dim=0)

    m = (unlearned_model_preds + original_model_preds) / 2
    js = 0.5 * F.kl_div(
        torch.log(unlearned_model_preds), m, reduction='batchmean'
    ) + 0.5 * F.kl_div(torch.log(original_model_preds), m, reduction='batchmean')
    return js.item()


def membership_inference(model, retain_set, test_set, forget_set, random_state: int = 0, batch_size=64,
                         attacker='catboost',
                         verbose=True,
                         full_verbose=False,
                         device='cuda'):
    retain_counts, test_counts = {}, {}
    retain_set_per_class, test_set_per_class = {}, {}
    for sample in retain_set:
        x, y = sample
        if (int(y)) not in retain_counts:
            retain_counts[int(y)] = 0
            retain_set_per_class[int(y)] = []
        retain_counts[int(y)] += 1
        retain_set_per_class[int(y)].append(sample)
    for sample in test_set:
        x, y = sample
        if int(y) not in test_counts:
            test_counts[int(y)] = 0
            test_set_per_class[int(y)] = []
        test_counts[int(y)] += 1
        test_set_per_class[int(y)].append(sample)

    target_counts = {
        key: min(
            retain_counts.get(key, 0), test_counts.get(key, 0)
        ) for key in set(retain_counts) | set(test_counts)
    }
    sampled_retain_set, sampled_test_set = {}, {}
    random.seed(random_state)
    for k in target_counts:
        if target_counts[k] == 0:
            continue
        class_count = target_counts[k]
        sampled_retain_set[k] = random.sample(retain_set_per_class[k], min(class_count, len(retain_set_per_class[k])))
        sampled_test_set[k] = random.sample(test_set_per_class[k], min(class_count, len(test_set_per_class[k])))

    sampled_retain_set = [item for sublist in sampled_retain_set.values() for item in sublist]
    sampled_test_set = [item for sublist in sampled_test_set.values() for item in sublist]

    retain_loader = DataLoader(sampled_retain_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(sampled_test_set, batch_size=batch_size, shuffle=False)
    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=False)

    model.eval()
    mia_retain, mia_test, mia_forget = [], [], []
    with torch.no_grad():
        for x, _ in retain_loader:
            x = x.to(device)
            out = model(x)
            mia_retain.append(F.softmax(out, dim=-1).data)
        mia_retain = torch.cat(mia_retain)
        for x, _ in test_loader:
            x = x.to(device)
            out = model(x)
            mia_test.append(F.softmax(out, dim=-1).data)
        mia_test = torch.cat(mia_test)
        for x, _ in forget_loader:
            x = x.to(device)
            out = model(x)
            mia_forget.append(F.softmax(out, dim=-1).data)
        mia_forget = torch.cat(mia_forget)

    X_train = torch.cat([mia_retain, mia_test]).cpu().numpy()
    y_train = np.concatenate([np.ones(len(mia_retain)), np.zeros(len(mia_test))]).astype(int)

    X_test = mia_forget.cpu().numpy()
    y_test = np.concatenate([np.zeros(len(mia_forget))]).astype(int)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    if attacker == "catboost":
        clf = CatBoostClassifier(verbose=0, random_state=42)
    else:
        raise ValueError
    kf = StratifiedKFold(n_splits=5)
    metrics_train = {
        'Accuracy': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [], 'TPR': [], 'TNR': []
    }
    metrics_val = {
        'Accuracy': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [], 'TPR': [], 'TNR': []
    }
    metrics_test = {
        'TN': [], 'FP': [], 'TNR': []
    }
    if verbose:
        print("\n\n++++MIA++++")
    # cross-val
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        # fit the model
        clf.fit(X_train_fold, y_train_fold)

        # evaluate
        y_pred_train = clf.predict(X_train_fold)
        acc_train = accuracy_score(y_train_fold, y_pred_train)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_fold, y_pred_train).ravel()
        tpr_train = tp_train / (tp_train + fn_train)
        tnr_train = tn_train / (tn_train + fp_train)
        metrics_train['Accuracy'].append(acc_train)
        metrics_train['TP'].append(tp_train)
        metrics_train['TN'].append(tn_train)
        metrics_train['FP'].append(fp_train)
        metrics_train['FN'].append(fn_train)
        metrics_train['TPR'].append(tpr_train)
        metrics_train['TNR'].append(tnr_train)

        y_pred_val = clf.predict(X_val_fold)
        acc_val = accuracy_score(y_val_fold, y_pred_val)
        tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_fold, y_pred_val).ravel()
        tpr_val = tp_val / (tp_val + fn_val)
        tnr_val = tn_val / (tn_val + fp_val)
        metrics_val['Accuracy'].append(acc_val)
        metrics_val['TP'].append(tp_val)
        metrics_val['TN'].append(tn_val)
        metrics_val['FP'].append(fp_val)
        metrics_val['FN'].append(fn_val)
        metrics_val['TPR'].append(tpr_val)
        metrics_val['TNR'].append(tnr_val)

        y_pred_test = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        # Check the shape of the confusion matrix and unpack
        if conf_matrix.shape == (1, 1):
            # Only one class present in y_test and y_pred_test
            tn_test = conf_matrix[0, 0] if y_test[0] == 0 else 0
            fp_test = 0
        elif conf_matrix.shape == (2, 2):
            # Both classes are present or predicted
            tn_test, fp_test, _, _ = conf_matrix.ravel()
        else:
            raise ValueError("Unexpected shape of confusion matrix")
        tnr_test = tn_test / (tn_test + fp_test)
        metrics_test['TN'].append(tn_test)
        metrics_test['FP'].append(fp_test)
        metrics_test['TNR'].append(tnr_test)

        if full_verbose:
            print(f"Fold {fold} Training Set Metrics:")
            print(f"Accuracy: {acc_train}, TP: {tp_train}, TN: {tn_train}, FP: {fp_train}, FN: {fn_train},"
                  f" TPR: {tpr_train}, TNR: {tnr_train}")
            print(f"Fold {fold} Validation Set Metrics:")
            print(f"Accuracy: {acc_val}, TP: {tp_val}, TN: {tn_val}, FP: {fp_val}, FN: {fn_val}, "
                  f"TPR: {tpr_val}, TNR: {tnr_val}")
            print(f"Fold {fold} Test Set Metrics:")
            print(f"TN: {tn_test}, FP: {fp_test}, TNR: {tnr_test}")

    # Calculate average of metrics for cross-validation and test set
    avg_metrics_cv = {k: np.mean(v) for k, v in metrics_val.items()}
    avg_metrics_test = {k: np.mean(v) for k, v in metrics_test.items()}
    if verbose:
        print("Average Validation Metrics:")
        print(f"{avg_metrics_cv}")
        print("++++MIA++++\n\n")
    return avg_metrics_cv, avg_metrics_test

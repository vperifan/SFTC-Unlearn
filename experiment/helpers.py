import copy
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Tuple, Union, List, Dict, Callable
import json

import pandas as pd
import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, efficientnet_b0

from utils.preprocessing import pre_process_cifar10, to_torch_loader, pre_process_korean_family, pre_process_fer

from utils.unlearning_utils import UnLearningData, RandomDistributionGenerator, CustomPseudoLabelDataset

from utils.unlearning_alg.simple_finetuning import fine_tune
from utils.unlearning_alg.cf_eu_k import catastrophic_forgetting_k
from utils.unlearning_alg.neg_grad import neg_grad
from utils.unlearning_alg.scrub import scrub
from utils.unlearning_alg.bad_teaching import bad_teaching
from utils.unlearning_alg.sftc_unlearn import sftc_unlearn


def get_data_training(
        args, return_sets: bool = False
) -> Union[Tuple[DataLoader, DataLoader, DataLoader, DataLoader], Tuple[
    DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]
]:
    try:
        note = args.note
    except KeyError:
        note = False
    except AttributeError:
        note = False
    try:
        unlearn = args.unlearn
    except KeyError:
        unlearn = False
    except AttributeError:
        unlearn = False
    training = False if unlearn else True
    dataset = args.dataset.lower()
    if dataset == "cifar":
        train_set, val_set, test_set, retain_set, forget_set = pre_process_cifar10(
            rng=None,
            training=training,
            note=note
        )
    elif dataset == "imbalanced_cifar":
        train_set, val_set, test_set, retain_set, forget_set = pre_process_cifar10(
            rng=None,
            imbalanced=True,
            training=training,
            note=note
        )
    elif dataset == "mufac":
        train_set, val_set, test_set, retain_set, forget_set = pre_process_korean_family(
            rng=None,
            training=training,
            note=note
        )
    elif args.dataset.lower() == "fer":
        train_set, val_set, test_set, retain_set, forget_set = pre_process_fer(
            rng=None,
            training=training,
            note=note
        )
    else:
        raise ValueError

    if return_sets:
        return retain_set, val_set, test_set, forget_set

    try:
        unlearn_alg = args.algorithm
    except KeyError:
        unlearn_alg = None
    except AttributeError:
        unlearn_alg = None
    if unlearn_alg is not None and unlearn_alg == "bad_teaching":
        retain_set = UnLearningData(forget_data=forget_set, retain_data=retain_set)

    test_loader = to_torch_loader(test_set, batch_size=args.batch_size, shuffle=False)
    val_loader = to_torch_loader(val_set, batch_size=args.batch_size, shuffle=False)
    retain_loader = to_torch_loader(retain_set, batch_size=args.batch_size, shuffle=True)
    train_loader = to_torch_loader(train_set, batch_size=args.batch_size, shuffle=True)
    forget_loader = to_torch_loader(forget_set, batch_size=args.batch_size, shuffle=True)

    if unlearn_alg is not None and unlearn_alg == "sftc":
        retain_set_pseudo = CustomPseudoLabelDataset(retain_set, 0)
        forget_set_pseudo = CustomPseudoLabelDataset(forget_set, 1)
        merged_set = ConcatDataset([retain_set_pseudo, forget_set_pseudo])
        merged_loader = to_torch_loader(merged_set, batch_size=args.batch_size, shuffle=True)
        return retain_loader, val_loader, test_loader, forget_loader, merged_loader

    if args.mode.lower() == "full":
        return train_loader, val_loader, test_loader, forget_loader
    else:
        return retain_loader, val_loader, test_loader, forget_loader


def get_model(args) -> torch.nn.Module:
    if args.dataset.lower() == "cifar" or args.dataset.lower() == "imbalanced_cifar":
        num_classes = 10
    elif args.dataset.lower() == "mufac":
        num_classes = 8
    elif args.dataset.lower() == "fer":
        num_classes = 7
    else:
        raise ValueError
    if args.model.lower() == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        if args.dataset.lower() == "fer":
            original_conv = model.conv1
            model.conv1 = torch.nn.Conv2d(1, original_conv.out_channels,
                                          kernel_size=original_conv.kernel_size,
                                          stride=original_conv.stride,
                                          padding=original_conv.padding, bias=False)
    elif args.model.lower() == "efficientnet":
        model = efficientnet_b0(weights=None, num_classes=num_classes)
        if args.dataset.lower() == "fer":
            original_first_layer = model.features[0][0]
            model.features[0][0] = torch.nn.Conv2d(1, original_first_layer.out_channels,
                                                   kernel_size=original_first_layer.kernel_size,
                                                   stride=original_first_layer.stride,
                                                   padding=original_first_layer.padding,
                                                   bias=original_first_layer.bias)

    else:
        raise ValueError
    return model


def get_dummy_model(args) -> Union[torch.nn.Module, RandomDistributionGenerator]:
    if args.dummy_model.lower() == "random":
        if args.dataset.lower() == "cifar" or args.dataset.lower() == "imbalanced_cifar":
            num_classes = 10
        elif args.dataset.lower() == "mufac":
            num_classes = 8
        elif args.dataset.lower() == "fer":
            num_classes = 7
        else:
            raise ValueError
        dummy_model = RandomDistributionGenerator(dist='normal', dimensions=num_classes)
        return dummy_model
    elif args.dummy_model.lower() == "model":
        return get_model(args)


def get_pretrained_model(args, path=None) -> torch.nn.Module:
    if args.dataset.lower() == "cifar" or args.dataset.lower() == "imbalanced_cifar":
        num_classes = 10

    elif args.dataset.lower() == "mufac":
        num_classes = 8
    elif args.dataset.lower() == "fer":
        num_classes = 7
    else:
        raise ValueError

    # Check if the specified file path exists
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The {path} file does not exists!")
    else:
        # Check if the 'checkpoints' folder exists
        if not os.path.exists('checkpoints'):
            if not os.path.exists('../checkpoints'):
                raise FileNotFoundError("The 'checkpoints' folder does not exists! Please train a model first!")
            else:
                chk_path = '../checkpoints'
        else:
            chk_path = 'checkpoints'
        # Check if the dataset folder exists inside 'checkpoints'
        dataset_folder_path = os.path.join(chk_path, args.dataset)
        if not os.path.exists(dataset_folder_path):
            raise FileNotFoundError(f"The folder for dataset '{args.dataset}' does not exist in 'checkpoints'.")
        # Find the first .pth file in the dataset folder
        pth_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.pth') and args.model.lower() in f]
        if not pth_files:
            raise FileNotFoundError("No .pth files found in the dataset folder.")
        pth_files = [f for f in pth_files if "retain" not in f]
        print(f"Using the {pth_files[0]} model.")
        path = os.path.join(dataset_folder_path, pth_files[0])

    if args.model.lower() == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        if args.dataset.lower() == "fer":
            original_conv = model.conv1
            model.conv1 = torch.nn.Conv2d(1, original_conv.out_channels,
                                          kernel_size=original_conv.kernel_size,
                                          stride=original_conv.stride,
                                          padding=original_conv.padding, bias=False)
    elif args.model.lower() == "efficientnet":
        model = efficientnet_b0(weights=None, num_classes=num_classes)
        if args.dataset.lower() == "fer":
            original_first_layer = model.features[0][0]
            model.features[0][0] = torch.nn.Conv2d(1, original_first_layer.out_channels,
                                                   kernel_size=original_first_layer.kernel_size,
                                                   stride=original_first_layer.stride,
                                                   padding=original_first_layer.padding,
                                                   bias=original_first_layer.bias)
    else:
        raise ValueError

    weights_pretrained = torch.load(path)
    model.load_state_dict(weights_pretrained)
    return model


def get_pretrained_models(args, full: bool = True) -> List[torch.nn.Module]:
    if args.dataset.lower() == "cifar" or args.dataset.lower() == "imbalanced_cifar":
        num_classes = 10

    elif args.dataset.lower() == "mufac":
        num_classes = 8
    elif args.dataset.lower() == "fer":
        num_classes = 7
    else:
        raise ValueError

    # Check if the 'checkpoints' folder exists
    if not os.path.exists('checkpoints'):
        if not os.path.exists('../checkpoints'):
            raise FileNotFoundError("The 'checkpoints' folder does not exists! Please train a model first!")
        else:
            chk_path = '../checkpoints'
    else:
        chk_path = 'checkpoints'
    # Check if the dataset folder exists inside 'checkpoints'
    dataset_folder_path = os.path.join(chk_path, args.dataset)
    if not os.path.exists(dataset_folder_path):
        raise FileNotFoundError(f"The folder for dataset '{args.dataset}' does not exist in 'checkpoints'.")
    # Find the first .pth file in the dataset folder
    pth_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.pth') and args.model.lower() in f]
    if not pth_files:
        raise FileNotFoundError("No .pth files found in the dataset folder.")
    if full:
        model_type = 'full'
    else:
        model_type = 'retain'
    pth_files = [f for f in pth_files if model_type in f]

    if args.model.lower() == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        if args.dataset.lower() == "fer":
            original_conv = model.conv1
            model.conv1 = torch.nn.Conv2d(1, original_conv.out_channels,
                                          kernel_size=original_conv.kernel_size,
                                          stride=original_conv.stride,
                                          padding=original_conv.padding, bias=False)
    elif args.model.lower() == "efficientnet":
        model = efficientnet_b0(weights=None, num_classes=num_classes)
        if args.dataset.lower() == "fer":
            original_first_layer = model.features[0][0]
            model.features[0][0] = torch.nn.Conv2d(1, original_first_layer.out_channels,
                                                   kernel_size=original_first_layer.kernel_size,
                                                   stride=original_first_layer.stride,
                                                   padding=original_first_layer.padding,
                                                   bias=original_first_layer.bias)
    else:
        raise ValueError

    models = []
    for path in pth_files:
        path = os.path.join(dataset_folder_path, path)
        weights_pretrained = torch.load(path)
        model = copy.deepcopy(model)
        model.load_state_dict(weights_pretrained)
        models.append(model)
    print(pth_files)
    return models


def get_optimizer(args, model) -> torch.optim.Optimizer:
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.decay
        )
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=args.momentum
        )
    else:
        raise ValueError
    return optimizer


def get_scheduler(args, optimizer) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.epochs, verbose=True
        )
        return scheduler
    return None


def calculate_class_weights(data_loader, log_count=True):
    """Calculates class weights based on the frequency of each class."""
    # get the labels from the training set
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    if log_count:
        print(f"Counts per class: {np.bincount(all_labels)}")

    # calculate class weights
    classes = np.unique(all_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)

    return torch.tensor(class_weights, dtype=torch.float)


def get_criterion(args, data_loader, device) -> torch.nn.Module:
    if args.class_weights:
        class_weights = calculate_class_weights(data_loader)
        class_weights = class_weights.to(device)
        print(f"Class Weights: {class_weights}")
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion


def get_unlearning_algorithm(args) -> Callable:
    if args.algorithm.lower() == "finetuning":
        return fine_tune
    if args.algorithm.lower() == "cfk" or args.algorithm.lower() == "euk":
        return catastrophic_forgetting_k
    elif args.algorithm.lower() == "neg_grad" or args.algorithm.lower() == "advanced_neg_grad":
        return neg_grad
    elif args.algorithm.lower() == "scrub":
        return scrub
    elif args.algorithm.lower() == "bad_teaching":
        return bad_teaching
    elif args.algorithm.lower() == "sftc":
        return sftc_unlearn
    else:
        raise ValueError


def store_training_history(args, history: List[Dict[str, List[float]]], name):
    losses, accs = history
    train_losses, val_losses, test_losses = losses['train'], losses['val'], losses['test']
    try:
        train_accs = accs['train']
    except KeyError:
        train_accs = []
    val_accs, test_accs = accs['val'], accs['test']
    try:
        forget_losses = losses['forget']
        forget_accs = accs['forget']
    except KeyError:
        forget_losses, forget_accs = None, None
    # Create a 2x1 grid of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5))

    # Subplot 1: Losses
    if len(train_losses) > 0:
        ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    ax1.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', marker='o')
    if forget_losses is not None:
        ax1.plot(range(1, len(forget_losses) + 1), forget_losses, label='Forget Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss per Epoch')
    ax1.legend()

    # Subplot 2: Accuracies
    if len(train_accs) > 0:
        ax2.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy', marker='o')
    ax2.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', marker='o')
    ax2.plot(range(1, len(test_accs) + 1), test_accs, label='Test Accuracy', marker='o')
    if forget_accs is not None:
        ax2.plot(range(1, len(forget_accs) + 1), forget_accs, label='Forget Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.legend()

    plt.tight_layout()

    if forget_losses is not None:
        base_path = 'unlearn_history'
    else:
        base_path = 'history'
    # Check if 'history' folder exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # Check if dataset specific folder exists inside 'history'
    dataset_folder = os.path.join(base_path, str(args.dataset))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    # Save the figure
    fig.savefig(os.path.join(dataset_folder, name + '.png'))

    # store history to csv
    epochs = range(1, len(test_losses) + 1)

    data = {
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Test Loss': test_losses,
        'Train Acc': train_accs,
        'Val Acc': val_accs,
        'Test Acc': test_accs
    }

    if len(train_accs) == 0:
        del data['Train Acc']
    if len(train_losses) == 0:
        del data['Train Loss']

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(dataset_folder, name + '.csv'), index=False)


def store_trained_model(args, model, name, unlearn=False):
    if unlearn:
        base_path = 'unlearn_checkpoints'
    else:
        base_path = 'checkpoints'
    # Check if 'checkpoints' folder exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # Check if dataset specific folder exists inside 'checkpoints'
    dataset_folder = os.path.join(base_path, str(args.dataset))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Save the model
    model_path = os.path.join(dataset_folder, name + '.pth')
    torch.save(model.state_dict(), model_path)

    # Save the arguments
    args_path = os.path.join(dataset_folder, name + '_args.json')
    with open(args_path, 'w') as f:
        # Convert args to a dictionary if it's not already in that format
        args_dict = vars(args) if hasattr(args, '__dict__') else args
        json.dump(args_dict, f, indent=4)

import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Union

import copy

import pandas as pd
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset


def pre_process_cifar10(
        rng,
        training=False,
        imbalanced=False,
        note=False,
):
    if rng is None:
        rng = torch.Generator().manual_seed(42)

    if note:
        pth = "../data"
    else:
        pth = "./data"

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if training:
        train_normalize = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        train_normalize = normalize

    train_set = torchvision.datasets.CIFAR10(
        root=pth, train=True, download=True, transform=train_normalize
    )

    held_out = torchvision.datasets.CIFAR10(
        root=pth, train=False, download=True, transform=normalize
    )

    if imbalanced:
        original_train_set = copy.deepcopy(train_set)
        original_held_out = copy.deepcopy(held_out)

        local_path = f"{pth}/imbalanced_train_idx_cifar.npy"
        train_idx = np.load(local_path)
        local_path = f"{pth}/imbalanced_val_idx_cifar.npy"
        val_idx = np.load(local_path)
        local_path = f"{pth}/imbalanced_test_idx_cifar.npy"
        test_idx = np.load(local_path)
        local_path = f"{pth}/imbalanced_forget_idx_cifar.npy"
        forget_idx = np.load(local_path)

        train_set = torch.utils.data.Subset(original_train_set, train_idx)
        val_set = torch.utils.data.Subset(original_held_out, val_idx)
        test_set = torch.utils.data.Subset(original_held_out, test_idx)
        forget_set = torch.utils.data.Subset(original_train_set, forget_idx)

        # Create a boolean mask for forgetting
        forget_mask = np.zeros(len(original_train_set.targets), dtype=bool)
        forget_mask[forget_idx] = True
        # Invert the mask to get indices to retain
        retain_mask = ~forget_mask
        # Apply the retain mask to train_idx
        retain_train_idx = np.array(train_idx)[retain_mask[train_idx]]
        # Create the retain set and updated train set
        retain_set = torch.utils.data.Subset(original_train_set, retain_train_idx)


    else:
        # split held out data into validation and test set
        test_set, val_set = torch.utils.data.random_split(
            held_out, [0.5, 0.5], generator=rng
        )

        # download the forget and retain index split
        try:
            local_path = "./data/forget_idx_cifar.npy"
            forget_idx = np.load(local_path)
        except:
            local_path = "../data/forget_idx_cifar.npy"
            forget_idx = np.load(local_path)

        # construct indices of retain set from those of the forget set
        forget_mask = np.zeros(len(train_set.targets), dtype=bool)
        forget_mask[forget_idx] = True
        retain_idx = np.arange(forget_mask.size)[~forget_mask]

        # split train set into a forget and a retain set
        forget_set = torch.utils.data.Subset(train_set, forget_idx)
        retain_set = torch.utils.data.Subset(train_set, retain_idx)

    print(f"#Training: {len(train_set)}")
    print(f"#Validation: {len(val_set)}")
    print(f"#Test: {len(test_set)}")
    print(f"#Retain: {len(retain_set)}")
    print(f"#Forget: {len(forget_set)}")

    print(train_normalize)

    return train_set, val_set, test_set, retain_set, forget_set

class FERDataset(Dataset):

    def __init__(self, images, labels, transforms):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if i < 0 or i >= len(self.X):
            print(f"__getitem__ called with out-of-range index {i}. Dataset size is {len(self.X)}")
            raise IndexError("Index out of range")
        data = [int(m) for m in self.X[i].split(' ')]
        data = np.asarray(data).astype(np.uint8).reshape(48, 48, 1)
        data = self.transforms(data)
        label = self.y[i]
        return data, label


def pre_process_fer(
        rng,
        training=False,
        note=False,
):
    if rng is None:
        rng = torch.Generator().manual_seed(42)

    if note:
        pth = "../data/fer/fer2013/fer2013.csv"
        forget_pth = "../data/fer_forget_idx.npy"
    else:
        pth = "./data/fer/fer2013/fer2013.csv"
        forget_pth = "./data/fer_forget_idx.npy"

    # pre-process FER
    normalize = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
    )
    if training:
        train_normalize = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(128),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(128),# padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        )
    else:
        train_normalize = normalize

    df = pd.read_csv(pth)
    df_train = df[df.Usage == 'Training'].drop(['Usage'], axis=1).reset_index().drop(columns=['index'], axis=1)
    df_val = df[df.Usage == 'PublicTest'].drop(['Usage'], axis=1).reset_index().drop(columns=['index'], axis=1)
    df_test = df[df.Usage == 'PrivateTest'].drop(['Usage'], axis=1).reset_index().drop(columns=['index'], axis=1)

    train_images = df_train.iloc[:, 1]
    train_labels = df_train.iloc[:, 0]
    val_images = df_val.iloc[:, 1]
    val_labels = df_val.iloc[:, 0]
    test_images = df_test.iloc[:, 1]
    test_labels = df_test.iloc[:, 0]

    forget_idx = np.load(forget_pth)
    df_forget = df_train.loc[forget_idx].reset_index().drop(columns=['index'], axis=1)
    df_retain = df_train[~df_train.index.isin(forget_idx)].reset_index().drop(columns=['index'], axis=1)
    forget_images = df_forget.iloc[:, 1]
    forget_labels = df_forget.iloc[:, 0]
    retain_images = df_retain.iloc[:, 1]
    retain_labels = df_retain.iloc[:, 0]

    # assigning the transformed data
    train_set = FERDataset(train_images, train_labels, train_normalize)
    val_set = FERDataset(val_images, val_labels, normalize)
    test_set = FERDataset(test_images, test_labels, normalize)
    retain_set = FERDataset(retain_images, retain_labels, train_normalize)
    forget_set = FERDataset(forget_images, forget_labels, train_normalize)

    print(f"#Training: {len(train_set)}")
    print(f"#Validation: {len(val_set)}")
    print(f"#Test: {len(test_set)}")
    print(f"#Retain: {len(retain_set)}")
    print(f"#Forget: {len(forget_set)}")
    print(train_normalize)

    return train_set, val_set, test_set, retain_set, forget_set

def parsing(meta_data):
    image_age_list = []
    # iterate all rows in the metadata file
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list


class KoreanDataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None, forget=False, retain=False):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata.
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label


def pre_process_korean_family(
        rng,
        training=False,
        note=False
):
    if rng is None:
        rng = torch.Generator().manual_seed(42)

    if note:
        pth = "../data"
    else:
        pth = "./data"

    data_path = os.path.join(pth, "custom_korean_family_dataset_resolution_128")
    if not os.path.exists(data_path):
        filename = os.path.join(pth, "custom_korean_family_dataset_resolution_128.zip")
        url = "https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg?download=1"
        command = ['wget', '-O', filename, url]
        subprocess.run(command)
        with zipfile.ZipFile(filename, 'r') as zip_file:
            zip_file.extractall(data_path)

    train_meta_data_path = os.path.join(data_path, "custom_train_dataset.csv")
    train_meta_data = pd.read_csv(train_meta_data_path)
    train_image_directory = os.path.join(data_path, "train_images")

    val_meta_data_path = os.path.join(data_path, "custom_val_dataset.csv")
    val_meta_data = pd.read_csv(val_meta_data_path)
    val_image_directory = os.path.join(data_path, "val_images")

    unseen_meta_data_path = os.path.join(data_path, "custom_test_dataset.csv")
    unseen_meta_data = pd.read_csv(unseen_meta_data_path)
    unseen_image_directory = os.path.join(data_path, "test_images")

    normalize = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])

    if training:
        train_normalize = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])

    else:
        train_normalize = normalize
    # train_normalize = normalize
    train_set = KoreanDataset(train_meta_data, train_image_directory, train_normalize)
    retain_set = KoreanDataset(train_meta_data, train_image_directory, train_normalize, retain=True)
    forget_set = KoreanDataset(train_meta_data, train_image_directory, train_normalize, forget=True)
    val_set = KoreanDataset(val_meta_data, val_image_directory, normalize)
    test_set = KoreanDataset(unseen_meta_data, unseen_image_directory, normalize)

    print(f"#Training: {len(train_set)}")
    print(f"#Validation: {len(val_set)}")
    print(f"#Test: {len(test_set)}")
    print(f"#Retain: {len(retain_set)}")
    print(f"#Forget: {len(forget_set)}")
    print(train_normalize)

    return train_set, val_set, test_set, retain_set, forget_set


def to_torch_loader(
        data,
        batch_size: int = 128,
        seed: Union[int, None] = None,
        shuffle: bool = False
) -> DataLoader:
    """
    Create a PyTorch DataLoader with customizable batch size, shuffling, and optional seed for reproducibility.

    Args:
        data: Dataset to be loaded.
        batch_size (int): Batch size for the DataLoader.
        seed (Union[int, None]): Random seed for shuffling. If None, a random seed is generated.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    if shuffle:
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
            print(f"Using seed={seed}")
        data_generator = torch.Generator().manual_seed(seed)
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=data_generator,
            num_workers=2,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    return data_loader

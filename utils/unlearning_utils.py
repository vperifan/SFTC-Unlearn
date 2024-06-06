import random
from typing import Union, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomPseudoLabelDataset(Dataset):
    """
    A custom dataset that includes pseudo labels indicating whether a sample belongs to the retain set or not.
    """

    def __init__(self, dataset, pseudo_label):
        self.dataset = dataset
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, self.pseudo_label


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = self.forget_data[index][1]
            pseudo_label = 1
            return x, y, pseudo_label
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = self.retain_data[index - self.forget_len][1]
            pseudo_label = 0
            return x, y, pseudo_label


class RandomDistributionGenerator:
    def __init__(self, dist, dimensions: int = 10):
        distributions = {
            'normal': lambda size: np.abs(np.random.normal(size=size)),
            'poisson': lambda size: np.abs(np.random.poisson(size=size)),
            'uniform': lambda size: np.abs(np.random.uniform(size=size))
        }
        if dist not in distributions:
            raise ValueError(f'Unknown distribution: {dist}')

        self.dist_fn = distributions[dist]
        self.dimensions = dimensions

    def __call__(self, num_gen: int, labels: Union[torch.Tensor, List[int], np.ndarray, None],
                 *args, **kwargs):
        values = self.dist_fn(size=(num_gen, self.dimensions))

        if labels is not None:
            assert num_gen == len(labels)
            for i, label in enumerate(labels):
                while np.argmax(values[i]) != label:
                    values[i, label] += random.uniform(0.5, 1)

        values = torch.from_numpy(values)
        if isinstance(labels, torch.Tensor):
            values = values.to(labels.device)
        return values

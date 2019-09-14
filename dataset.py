import os

import numpy as np

import torch
from torch.utils.data import Dataset


class Snelson(Dataset):
    def __init__(self, root, n=100, test_x=True, x_min=None, x_max=None, n_test=None, normalize=False):
        self.root = os.path.join(root, 'snelson') if 'snelson' not in root else root

        self.n = n
        self.train_x, self.train_y, self.test_x = self.load_snelson_data()
        self.train_x, self.train_y = self.train_x[:n], self.train_y[:n]

        if test_x:
            self.x_min = torch.min(self.test_x)
            self.x_max = torch.max(self.test_x)
            self.n_test = self.test_x.size(0)
        else:
            self.x_min = x_min
            self.x_max = x_max
            self.n_test = n_test
            self.test_x = torch.from_numpy(
                np.linspace(x_min, x_max, num=n_test,
                            dtype=np.float32)).unsqueeze(-1)
        if normalize:
            self.normalize()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.train_x[idx], self.train_y[idx]

    @property
    def range(self):
        return self.x_min, self.x_max

    def load_snelson_data(self):
        train_x = self._load_snelson('train_inputs')
        train_y = self._load_snelson('train_outputs')
        test_x = self._load_snelson('test_inputs')

        perm = np.random.permutation(train_x.shape[0])
        train_x = train_x[perm]
        train_y = train_y[perm]

        translate = torch.from_numpy

        return translate(train_x).unsqueeze(-1), translate(train_y).unsqueeze(
            -1), translate(test_x).unsqueeze(-1)

    def _load_snelson(self, filename):
        with open(os.path.join(self.root, filename), 'r') as f:
            return np.array([float(i) for i in f.read().strip().split("\n")], dtype=np.float32)

    def normalize(self):
        self.mean_x = torch.mean(self.train_x, dim=0, keepdim=True)
        self.std_x = torch.std(self.train_x, dim=0, keepdim=True) + 1e-6
        self.mean_y = torch.mean(self.train_y, dim=0, keepdim=True)
        self.std_y = torch.std(self.train_y, dim=0, keepdim=True) + 1e-6

        for x in [self.train_x, self.test_x]:
            x -= self.mean_x
            x /= self.std_x

        for x in [self.x_min, self.x_max]:
            x -= self.mean_x
            x /= self.std_x

        self.train_y -= self.mean_y
        self.train_y /= self.std_y
import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, *numpy, transform=None):
        assert all(numpy[0].shape[0] == npy.shape[0] for npy in numpy)
        self.numpy = numpy
        self.transform = transform
    def __len__(self):
        return self.numpy[0].shape[0]
    def __getitem__(self, index):
        x = self.numpy[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.numpy[1][index]
        return x, y, index
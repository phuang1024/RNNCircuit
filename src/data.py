import random

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset

from constants import *


class CircuitDataset(Dataset):
    """
    The dataset has a time series of data, and randomly takes sections of it.

    The dataset length is a fixed, arbitrary number. Use it to determine the
    number of steps per epoch.
    Every getitem will be a random selection of data.

    Data random parameters:
    - Start index.
    - Index step.

    Over training, the minimum length should increase.
    This allows the model to train quickly in the beginning, and more
    accurately in the end.
    Set this with self.progress
    """

    def __init__(self, data):
        """
        data: (N, 3) simulation data from read_data()
        """
        self.data = torch.tensor(data)
        self.total_len = self.data.shape[0]

        self.set_progress(0)

    def __len__(self):
        return DATASET_LENGTH

    def __getitem__(self, _):
        """
        return: (dt, x, y):
            dt, x, y: (N,)
        """
        start = random.randint(0, self.total_len - self.seq_len - 1)
        max_step = (self.total_len - start) // self.seq_len
        max_step = min(max_step, 6)
        step = random.randint(1, max_step)

        data = self.data[start : start + step * self.seq_len : step]
        assert data.shape[0] == self.seq_len

        t = data[:, 0]
        x = data[:, 1]
        y = data[:, 2]
        dt = torch.diff(t, prepend=t[0:1])

        return dt, x, y

    def set_progress(self, progress):
        """
        progress: 0 to 1, overall training progress.
        """
        self.seq_len = int(np.interp(progress ** 2, [0, 1], [MIN_LENGTH, self.total_len / 2]))


def read_data(file):
    """
    Read LTSpice simulation data.

    return: (t, input, output).
        np array shape (N, 3) dtype float32.
    """
    data = []
    with open(file, "r") as fp:
        lines = fp.read().strip().split("\n")
        lines = lines[1:]
        for line in lines:
            parts = line.strip().split("\t")
            parts = list(map(float, parts))
            data.append(parts)

    data = np.array(data, dtype=np.float32)

    return data


if __name__ == "__main__":
    data = read_data("sims/Rectifier_1.txt")
    print(data.shape)

    """
    t = data[:, 1]
    plt.plot(t)
    plt.show()
    """

    dataset = CircuitDataset(data)
    dataset.set_progress(0.5)

    # Plot samples
    dt, x, y = dataset[0]
    t = torch.cumsum(dt, dim=0)

    plt.plot(t.numpy(), x.numpy(), label="Input")
    plt.plot(t.numpy(), y.numpy(), label="Output")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("dataSample.png")

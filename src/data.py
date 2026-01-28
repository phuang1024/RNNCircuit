import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset

from constants import *


class CircuitDataset(Dataset):
    def __init__(self, data):
        """
        data: (N, 3) simulation data.
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0] - BPTT + 1

    def __getitem__(self, idx):
        """
        return: (x, y):
            x: (BPTT,)
            y: (1,)
        """
        data = self.data[idx : idx + BPTT]
        x = data[:, 2]
        y = data[-1, 1]

        return x, y


def read_data(file):
    """
    Read LTSpice simulation data.

    return: (N, 3) numpy array.
        Second dimension is (time, output, input).
    """
    data = []
    with open(file, "r") as fp:
        lines = fp.read().strip().split("\n")
        lines = lines[1:]
        for line in lines:
            parts = line.strip().split("\t")
            parts = list(map(float, parts))
            data.append(parts)

    data = np.array(data)
    return data


if __name__ == "__main__":
    data = read_data("SimData.txt")
    print(data.shape)

    t = data[:, 0]
    plt.plot(t)
    plt.show()

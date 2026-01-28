import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset

from constants import *


class CircuitDataset(Dataset):
    def __init__(self, data):
        """
        data: (N, 3) simulation data from read_data()
        """
        self.data = torch.tensor(data, dtype=torch.float32)

        # Convert time to dt.
        for i in range(self.data.shape[0] - 1):
            self.data[i, 0] = self.data[i + 1, 0] - self.data[i, 0]
        self.data[-1, 0] = 0

    def __len__(self):
        return self.data.shape[0] - BPTT

    def __getitem__(self, idx):
        """
        return: (dt, x, y):
            dt, x, y: (N,)
        """
        data = self.data[idx : idx + BPTT]
        dt = data[:, 0]
        x = data[:, 2]
        y = data[:, 1]

        return dt, x, y


def read_data(file, data_step):
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
    data = data[::DATA_STEP]
    return data


if __name__ == "__main__":
    data = read_data("SimData.txt", 1)
    print(data.shape)

    t = data[:, 0]
    plt.plot(t)
    plt.show()

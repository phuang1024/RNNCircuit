import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset

from constants import *


class CircuitDataset(Dataset):
    def __init__(self, data):
        """
        data: (N, 4) simulation data from read_data()
        """
        self.data = torch.tensor(data)

    def __len__(self):
        return self.data.shape[0]# - BPTT

    def __getitem__(self, idx):
        """
        return: (dt, x, y):
            dt, x, y: (N,)
        """
        # TODO: Testing with using only the first sample.
        # Using all samples requires figuring out h(0) to satisfy initial cond.

        #data = self.data[idx : idx + BPTT]
        data = self.data[:self.data.shape[0] // 4]
        dt = data[:, 0]
        x = data[:, 2]
        y = data[:, 3]

        return dt, x, y


def read_data(file, data_step):
    """
    Read LTSpice simulation data.

    return: (dt, t, input, output).
        np array shape (N, 4) dtype float32.
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
    data = data[::data_step]

    # Compute dt.
    dt = np.zeros([data.shape[0], 1], dtype=np.float32)
    for i in range(data.shape[0] - 1):
        dt[i] = data[i + 1, 0] - data[i, 0]

    data = np.concatenate([dt, data], axis=1)

    return data


if __name__ == "__main__":
    data = read_data("VoltageDiv.txt", 1)
    print(data.shape)

    t = data[:, 1]
    plt.plot(t)
    plt.show()

"""
Test the NN on a ground truth signal.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from constants import *
from data import read_data
from model import CircuitRNN


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data")
    args = parser.parse_args()

    model = CircuitRNN()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    """
    model.f_x.weight[:] = 1
    model.f_final.weight[:] = 5
    model.g.weight[:] = 1
    """

    data = read_data(args.data, DATA_STEP)

    dt = data[:, 0]
    x = data[:, 2]
    y = data[:, 3]
    pred = model(torch.tensor(dt).unsqueeze(0), torch.tensor(x).unsqueeze(0))
    pred = pred.squeeze(0).numpy()

    # Integrate dt
    t = np.zeros([len(dt)], dtype=np.float32)
    for i in range(len(dt) - 1):
        t[i + 1] = t[i] + dt[i]

    # Plot.
    plt.plot(t, x, label="Input")
    plt.plot(t, y, label="Ground Truth")
    plt.plot(t, pred, label="Prediction")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

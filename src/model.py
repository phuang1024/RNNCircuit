import torch
import torch.nn as nn

from constants import *


class CircuitRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(1, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        output, hidden = self.rnn(x)
        y = self.output(hidden[0])

        return y


if __name__ == "__main__":
    model = CircuitRNN()

    x = torch.randn(100, 32, 1)
    y = model(x)
    print(y.shape)

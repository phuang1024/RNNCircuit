import torch
import torch.nn as nn

from constants import *


class CircuitRNN(nn.Module):
    """
    The dynamical system as a whole is:
    input: x(t)
    output: y(t)
    hidden state: h(t)
    network: f(h(t), x(t)) -> h'(t)
        and g(h(t)) -> y(t)

    This class implements the CT RNN with extra layer.
    input: dt(t) and x(t)
    output: y(t)
    Implements an Euler diff eq solver.
    """

    def __init__(self):
        super().__init__()

        self.tanh = nn.Tanh()

        # Layers for "f" function.
        self.f_x = nn.Linear(1, INNER_SIZE)
        self.f_h = nn.Linear(HIDDEN_SIZE, INNER_SIZE)
        self.f_final = nn.Linear(INNER_SIZE, HIDDEN_SIZE)

        # Layers for the "g" function.
        self.g = nn.Linear(HIDDEN_SIZE, 1)

    def forward_f(self, x, h):
        """
        Implements the "f" function.
        """
        return -1 / TAU * x + self.tanh(self.f_final(self.f_x(x) + self.f_h(h)))

    def forward(self, dt, x):
        """
        Runs the RNN for an N step sequence, using Euler integration.
        dt, x: shape (B, N)
            dt is the time step size for each iteration.
        return: y shape (B, N)
        """
        B, N = dt.shape

        # Initialize hidden state to 0.
        h = torch.zeros([B, HIDDEN_SIZE])
        # Output tensor.
        y = torch.zeros([B, N])
        for i in range(dt.shape[1]):
            # Take current index of input.
            xi = x[:, i].unsqueeze(-1)
            # Compute f and g.
            h_prime = self.forward_f(xi, h)
            y[:, i] = self.g(h).squeeze(-1)

            # Apply h'
            h += h_prime * dt[:, i]

        return y


if __name__ == "__main__":
    model = CircuitRNN()

    dt = torch.full((32, 100), 0.01)
    x = torch.randn(32, 100)

    y = model(dt, x)
    print(y.shape)

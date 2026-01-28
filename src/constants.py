import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 32

# This is implemented as the L dimension of the input to the RNN.
BPTT = 100
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100

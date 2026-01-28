import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inner size is the intermediate size of the 2 layer neural network.
# Hidden size is the output of that network, which produces the hidden state.
INNER_SIZE = 64
HIDDEN_SIZE = 64
TAU = 100

# This is implemented as the N dimension of the input to the RNN.
BPTT = 100
BATCH_SIZE = 32
LR = 1e-2
LR_DECAY = 0.92
EPOCHS = 100

# Every nth sample.
DATA_STEP = 3

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from constants import *
from data import CircuitDataset, read_data
from model import CircuitRNN


def forward_dataset(model, loader, criterion, desc=""):
    pbar = tqdm(loader)
    for x, y in pbar:
        x = x.to(DEVICE).unsqueeze(-1).permute(1, 0, 2)
        y = y.to(DEVICE).unsqueeze(-1)
        pred = model(x)
        loss = criterion(pred, y)
        pbar.set_description(f"{desc}: loss={loss.item():.3f}")
        yield loss


def train():
    model = CircuitRNN().to(DEVICE)

    data = read_data("SimData.txt")
    dataset = CircuitDataset(data)
    train_len = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])
    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
    }
    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for loss in forward_dataset(model, train_loader, criterion, f"Train epoch {epoch}"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            for loss in forward_dataset(model, val_loader, criterion, f"Val epoch {epoch}"):
                pass


def main():
    train()


if __name__ == "__main__":
    main()

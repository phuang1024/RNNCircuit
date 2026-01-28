import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from constants import *
from data import CircuitDataset, read_data
from model import CircuitRNN


def forward_dataset(model, loader, criterion, desc=""):
    pbar = tqdm(loader)
    for dt, x, y in pbar:
        # x, y are (B, N)
        dt = dt.to(DEVICE)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(dt, x)
        loss = criterion(pred, y)

        pbar.set_description(f"{desc}: loss={loss.item():.3f}")
        yield loss


def train():
    model = CircuitRNN().to(DEVICE)

    data = read_data("VoltageDiv.txt", DATA_STEP)
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    writer = SummaryWriter()
    global_step = 0

    for epoch in range(EPOCHS):
        for loss in forward_dataset(model, train_loader, criterion, f"Train epoch {epoch}"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/Train", loss.item(), global_step)
            global_step += 1

        with torch.no_grad():
            total_loss = 0
            for loss in forward_dataset(model, val_loader, criterion, f"Val epoch {epoch}"):
                total_loss += loss.item()
            avg_loss = total_loss / len(val_loader)

            writer.add_scalar("Loss/Val", avg_loss, global_step)

        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
        scheduler.step()

        torch.save(model.state_dict(), "latest.pt")


def main():
    train()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

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


def train(args):
    model = CircuitRNN().to(DEVICE)
    model.init_weights()

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    data = read_data(args.data)
    dataset = CircuitDataset(data)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    writer = SummaryWriter(f"logs/{args.data.stem}")
    global_step = 0

    for epoch in range(EPOCHS):
        dataset.set_progress(epoch / EPOCHS)

        for loss in forward_dataset(model, data_loader, criterion, f"Train epoch {epoch}"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/Train", loss.item(), global_step)
            global_step += 1

        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
        scheduler.step()

        torch.save(model.state_dict(), f"{args.data.stem}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

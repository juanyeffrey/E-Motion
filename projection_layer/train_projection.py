# projection_layer/train_projection.py

import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from .config import DEVICE, BATCH_SIZE, LR, NUM_EPOCHS, WEIGHT_DECAY, DEFAULT_OUT_DIR
from .dataset import ProjectionDataset
from .model import ProjectionMLP


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss: minimize (1 - cos_sim).
    Both vectors are normalized before cosine.
    """
    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [batch]
    return 1.0 - cos_sim.mean()


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = cosine_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_hat = model(x)
        loss = cosine_loss(y_hat, y)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .npz file with training data")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset from {args.data} ...")
    dataset = ProjectionDataset(args.data)
    print(f"N={len(dataset)}, in_dim={dataset.in_dim}, out_dim={dataset.out_dim}")

    # Train/val split
    n_total = len(dataset)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ProjectionMLP(
        input_dim=dataset.in_dim,
        output_dim=dataset.out_dim,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = eval_epoch(model, val_loader)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(args.out_dir, "projection_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": dataset.in_dim,
                    "output_dim": dataset.out_dim,
                    "hidden_dim": args.hidden_dim,
                },
                best_ckpt_path,
            )
            print(f"  ↳ Saved new best checkpoint to {best_ckpt_path}")

    # Save training config/metadata
    meta = {
        "data": os.path.abspath(args.data),
        "out_dir": os.path.abspath(args.out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "best_val_loss": best_val_loss,
        "checkpoint": best_ckpt_path,
    }
    meta_path = os.path.join(args.out_dir, "projection_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

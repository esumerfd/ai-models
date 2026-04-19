"""
Training script for the Small Language Model.

Usage:
    python phase_1_training/train.py

Expects training text at the path defined by DATA_FILE in core/config.py.
Saves checkpoints to CHECKPOINT_DIR.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import (
    BATCH_SIZE, CONTEXT_LENGTH, STEPS,
    EVAL_INTERVAL, SAVE_INTERVAL, TRAIN_SPLIT,
    DATA_FILE, CHECKPOINT_DIR, LEARNING_RATE,
)
from core.tokenizer_utils import train_tokenizer, encode
from core.model import SmallLanguageModel, device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, data: list[int], context_length: int):
        self.data = torch.tensor(data, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def make_get_batch(train_loader, val_loader):
    """Return a get_batch(split) function backed by infinite iterators."""
    iters = {
        "train": iter(train_loader),
        "val": iter(val_loader),
    }

    def get_batch(split: str):
        nonlocal iters
        try:
            x, y = next(iters[split])
        except StopIteration:
            # Restart the iterator when the epoch ends
            iters[split] = iter(train_loader if split == "train" else val_loader)
            x, y = next(iters[split])
        return x, y

    return get_batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load text
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Training data not found at '{DATA_FILE}'. "
            "Run 'make retrieve' to fetch training data."
        )
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {DATA_FILE}")

    # 2. Train tokenizer
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tokenizer = train_tokenizer(text)
    vocab_size = len(tokenizer.get_vocab())

    # 3. Encode and split
    encoded = encode(tokenizer, text)
    print(f"Encoded to {len(encoded):,} tokens")

    train_size = int(len(encoded) * TRAIN_SPLIT)
    train_data = encoded[:train_size]
    val_data = encoded[train_size:]

    train_dataset = TextDataset(train_data, CONTEXT_LENGTH)
    val_dataset = TextDataset(val_data, CONTEXT_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train samples: {len(train_dataset):,}  Val samples: {len(val_dataset):,}")

    get_batch = make_get_batch(train_loader, val_loader)

    # 4. Build model
    model = SmallLanguageModel(vocab_size).to(device)
    print(f"Model parameters: {model.param_count():,}  Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=STEPS, pct_start=0.05
    )

    # 5. Training loop
    model.train()
    for step in range(STEPS):
        x, y = get_batch("train")
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_INTERVAL == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d}  Loss: {loss.item():.4f}  LR: {lr:.2e}")

        if step % SAVE_INTERVAL == 0:
            path = os.path.join(CHECKPOINT_DIR, f"model_step_{step}.pth")
            torch.save(model.state_dict(), path)
            print(f"           -> Saved checkpoint: {path}")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
